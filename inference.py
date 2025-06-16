import torch
import torch.nn as nn
from tokenizer import MultimodalTokenizer
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
#                                  Inference                                    #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Inference."
    
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
    
    vq_model.load_state_dict(torch.load("checkpoints/0000500.pt")['model'])
    vq_model.to(device)
    vq_model.eval()

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
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    
    start_time = time.time()
    embeddings_all = []

    logger.info(f"Training for {args.epochs} epochs...")
    # Initialize wandb
    with torch.no_grad():
        for idx, x in tqdm(enumerate(loader), total=len(loader)):
            inputs = x.to(device)
            #print(inputs)
            embeddings = vq_model(inputs)
            #print(embeddings)
            embeddings_all.append(embeddings)
    
    embeddings_all = torch.cat(embeddings_all, dim=0)
    embeddings_all = embeddings_all.cpu().numpy()
    import numpy as np
    np.save("embeddings_all.npy", embeddings_all)
    
    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='datasets/')

    parser.add_argument("--kg-path", type=str, default='datasets/primeKG/', help="path to the knowledge graph")
    parser.add_argument("--med-codes-pkg-map-path", type=str, default='datasets/code/all_codes_mapping_v2_w_og_mappings.parquet', help="path to the med codes package map")
    parser.add_argument("--graph-save-path", type=str, default='MedTok/graph/', help="path to save the graph")
    
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='MedTok/log/', help='please specify a cloud disk path, if not, local path')
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

    parser.add_argument("--codebook-size", type=int, default=21000, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=64, help="codebook dimension for graph quantization")
    parser.add_argument("--semantic-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--text-code-dim", type=int, default=64, help="codebook dimension for semantic quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")

    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=1024) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
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
