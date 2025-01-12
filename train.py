# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
#from dataset.augmentation import random_crop_arr
#from dataset.build import build_dataset
from tokenizer import MultimodalTokenizer
from dataset.toy_dataset_creator import MedCodeDataset, custom_collate_fn
from tqdm import tqdm
from loss import shared_loss, specific_loss



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.graph_model_name.replace("/", "-") + "_" + args.text_model_name.replace('/', "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        import json
        json.dump(vars(args), open(f"{experiment_dir}/args.json", "w"), indent=4)  # Save args to experiment folder
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = MultimodalTokenizer(
        text_model_name=args.text_model_name,
        graph_model_name=args.graph_model_name,
        graph_in_channels=args.graph_in_channels,
        graph_hidden_channels=args.graph_hidden_channels,
        graph_out_channels=args.graph_out_channels,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,  ##codebook for graph
        #semantic_code_dim=args.semantic_code_dim,    ##codebook for text
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        #dropout_p=args.dropout_p,
        #kmeans=args.kmeans,
    )
    logger.info(f"Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    print(vq_model.parameters())
    #logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    # Setup optimizer
    if not args.finetune_decoder:
        logger.info("Optimizing all parameters.")
        logger.info(f"no kmeans, args.lr = {args.lr}")
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        logger.info("Optimizing graph decoder.")
        optimizer = torch.optim.Adam(vq_model.decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


    # Setup data:
    dataset = MedCodeDataset(args.kg_path, args.graph_save_path, args.med_codes_pkg_map_path, args.text_model_name, max_length=512)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Dataset contains {len(dataset):,} medical codes ({args.data_path})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        model_state = checkpoint["model"]
        if args.finetune_decoder and args.enhanced_decoder:
            # if finetuning with enhanced decoder, you would expect the old shape not match
            try:
                # if you want to continue finetune the enhanced decoder
                missing, unexpected = vq_model.load_state_dict(model_state, strict=False)
                logger.info(f"Missing keys: {missing}")
                logger.info(f"Unexpected keys: {unexpected}")
            except:
                # if switching from small decoder to enhanced decoder, delete the old decoder keys first
                decoder_keys = [k for k in model_state.keys() if k.startswith("decoder.")]
                for k in decoder_keys:
                    del model_state[k]
                missing, unexpected = vq_model.load_state_dict(model_state, strict=False)
                logger.info(f"Missing keys: {missing}")
                logger.info(f"Unexpected keys: {unexpected}")
        else:
            vq_model.load_state_dict(model_state, strict=True)
            logger.info("Loaded model from checkpoint.")

        if args.ema:
            ema.load_state_dict(checkpoint["ema"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            logger.info("Optimizer starting from scratch.")
            
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    #vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    #vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()


    logger.info(f"Training for {args.epochs} epochs...")
    # Initialize wandb
    if rank == 0:
        import wandb
        wandb.init(project="MultimodalTokenizer", config=args, name=f"{time_record}-{model_string_name}")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for idx, x in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
            inputs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                quantized_result = vq_model(inputs)
                #loss = codebook_loss[0] + codebook_loss[1] + codebook_loss[2] + codebook_loss[3]
                codebook_loss = (quantized_result['shared_embed_loss'][0] + quantized_result['shared_embed_loss'][1] +
                                 quantized_result['text_specific_loss'][0] + quantized_result['text_specific_loss'][1] + 
                                 quantized_result['graph_specific_loss'][0]+quantized_result['graph_specific_loss'][1])
                
                shared_loss_11, shared_loss_12, shared_loss_21, shared_loss_22 = shared_loss(quantized_result['shared_text_embedding'], quantized_result['shared_graph_embedding'], quantized_result['text_feature'], quantized_result['graph_feature'])
                shared_loss_all = shared_loss_11 + shared_loss_12 + shared_loss_21 + shared_loss_22

                specific_loss_11, specific_loss_12, specific_loss_21, specific_loss_22 = specific_loss(z1 = quantized_result['specific_embedding_text'],
                                              z1_aug = quantized_result['specific_embedding_text_aug'],
                                              z2 = quantized_result['specific_embedding_graph'],
                                              z2_aug = quantized_result['specific_embedding_graph_aug'],
                                              z1_c = quantized_result['shared_text_embedding'],
                                              z2_c = quantized_result['shared_graph_embedding'])
                specific_loss_all = specific_loss_11 + specific_loss_12 + specific_loss_21 + specific_loss_22
                
                beta = args.commit-loss-beta
                lamb = args.specific-loss-lamb
                loss_common = codebook_loss + beta * shared_loss_all + lamb * specific_loss_all
        
            scaler.scale(loss_common).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            '''with torch.cuda.amp.autocast(dtype=ptdtype):
                quantized_result_star = vq_model(inputs)

                specific_loss_all = specific_loss(z1 = quantized_result['specific_embedding_text'],
                                              z1_aug = quantized_result_star['specific_embedding_text_aug'],
                                              z2 = quantized_result['specific_embedding_graph'],
                                              z2_aug = quantized_result_star['specific_embedding_graph_aug'],
                                              z1_c = quantized_result['shared_text_embedding'],
                                              z2_c = quantized_result['shared_graph_embedding'])
                loss_specific = specific_loss_all   
            optimizer.zero_grad()
            scaler.scale(loss_specific).backward()
            scaler.step(optimizer)
            scaler.update()'''

            # # Log loss values:
            loss = loss_common.item() #+ loss_specific.item()
            running_loss += loss
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                #codebook_loss = codebook_loss.item()
                #print(codebook_loss)
                loss_dict = {
                    'loss': loss,
                    'loss_common_all': shared_loss_all.item(),
                    'loss_common_11': shared_loss_11.item(),
                    'loss_common_12': shared_loss_12.item(),
                    'loss_common_21': shared_loss_21.item(),
                    'loss_common_22': shared_loss_22.item(),
                    'loss_specific_all': specific_loss_all.item(),
                    'loss_specific_11': specific_loss_11.item(),
                    'loss_specific_12': specific_loss_12.item(),
                    'loss_specific_21': specific_loss_21.item(),
                    'loss_specific_22': specific_loss_22.item(),
                    'vq_loss': codebook_loss.item(),
                    'vq_shared_loss': quantized_result['shared_embed_loss'][0].item(),
                    'vq_text_loss': quantized_result['text_specific_loss'][0].item(),
                    'vq_graph_loss': quantized_result['graph_specific_loss'][0].item(),
                    'commit_shared_loss': quantized_result['shared_embed_loss'][1].item(),
                    'commit_text_loss': quantized_result['text_specific_loss'][1].item(),
                    'commit_graph_loss': quantized_result['graph_specific_loss'][1].item(),
                    'codebook_usage_shared': quantized_result['shared_codebook_usage'],
                    'codebook_usage_text': quantized_result['text_specific_usage'],
                    'codebook_usage_graph': quantized_result['graph_specific_usage'],
                } 
                
                if rank == 0:
                    wandb.log({**loss_dict}, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{time_record}-{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                        # Get a list of all checkpoints in the directory
                        checkpoint_files = glob(os.path.join(formatted_time + "-" + checkpoint_dir, "*.pt"))
                        checkpoint_files.sort(key=os.path.getmtime)  # Sort by modification time

                        # Remove old checkpoints if there are more than max_checkpoints
                        while len(checkpoint_files) > args.max_checkpoints:
                            oldest_checkpoint = checkpoint_files.pop(0)  # Get the oldest checkpoint
                            os.remove(oldest_checkpoint)
                            print(f"Deleted old checkpoint: {oldest_checkpoint}")
                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()
            

    vq_model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    dist.destroy_process_group()
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='datasets/')

    parser.add_argument("--kg-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/primeKG/', help="path to the knowledge graph")
    parser.add_argument("--med-codes-pkg-map-path", type=str, default='/n/holylfs06/LABS/mzitnik_lab/Lab/shvat372/icml_paper/ICML_codes/graphs/all_codes_mappings_v3.parquet', help="path to the med codes package map")
    parser.add_argument("--graph-save-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/kg_temp_2912', help="path to save the graph")
    
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer/log/', help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--model", type=str, default="MultimodalTokenizer")
    parser.add_argument("--graph_model_name", type=str, choices=["GCN", "GAT", "GraphTransformer"], default="GCN")
    parser.add_argument("--text_model_name", type=str, choices=["bert-base-uncased"], default="bert-base-uncased")

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--graph_in_channels", type=int, default=64, help="input channels for graph encoder")
    parser.add_argument("--graph_hidden_channels", type=int, default=128, help="hidden channels for graph encoder") 
    parser.add_argument("--graph_out_channels", type=int, default=64, help="output channels for graph encoder")

    parser.add_argument("--codebook-size", type=int, default=30000, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=64, help="codebook dimension for graph quantization")
    parser.add_argument("--semantic-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--text-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--shared-loss-beta", type=float, default=0.5, help="shared loss beta in codebook loss")
    parser.add_argument("--specific-loss-lamb", type=float, default=0.5, help="specific loss lambda in codebook loss")

    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.2, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="/n/netscratch/mzitnik_lab/Lab/xsu/MultimodalTokenizer/pre_trained_model")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=1024) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) # better change to bf16 if GPU support

    parser.add_argument("--infer_interpolate", action='store_true', help="interpolate the positional encoding for higher resolution inference")
    parser.add_argument("--enhanced_decoder", action='store_true', help="whether using enhanced decoder")
    parser.add_argument("--kmeans", action='store_true', help="whether using kmeans for codebook initialization")
    parser.add_argument('--finetune_decoder', action='store_true', help='finetune decoder')
    args = parser.parse_args()
    main(args)