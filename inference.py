import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import time
import argparse

from MedTok.utils.distributed import init_distributed_mode
from MedTok.tokenizer import MultimodalTokenizer
from MedTok.dataset_creator import MedCodeDataset, custom_collate_fn
from tqdm import tqdm


#################################################################################
#                                  Inference                                    #
#################################################################################

def main(root, pre_trained_model_name):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Inference."
    
    args_path = f"{root}/{pre_trained_model_name}/args.json"
    checkpoint_path = f"{root}/{pre_trained_model_name}/checkpoints/0003000.pt"

    import json
    from argparse import Namespace
    params = json.load(open(args_path))
    args = Namespace(**params)

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

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
    
    vq_model.load_state_dict(torch.load(checkpoint_path)['model'])
    vq_model.to(device)
    vq_model.eval()

    print(vq_model.parameters())
    #logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")


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
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.eval()
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    
    start_time = time.time()
    embeddings_all = []
    tokens_all = []
    weights_all = []
    code_indices_all = []

    # Initialize wandb
    with torch.no_grad():
        for idx, x in tqdm(enumerate(loader), total=len(loader)):
            inputs = x.to(device)
            x_indices = x.code_indices
            #print(inputs)
            embeddings, tokens, weights = vq_model(inputs) ##embeddings (bz, dim), tokens(bz, 4, 5), weights(bz, 4, 5)
            #print(embeddings)
            embeddings_all.append(embeddings)
            code_indices_all.append(x_indices.float())
            tokens_all.append(tokens)
            weights_all.append(weights)
    
    print(code_indices_all)

    sorted_values_embeddings = [x for _, x in sorted(zip(x_indices, embeddings_all))]
    sorted_values_tokens = [x for _, x in sorted(zip(x_indices, tokens_all))]
    sorted_values_weights = [x for _, x in sorted(zip(x_indices, weights_all))]
    
    embeddings_all = torch.cat(sorted_values_embeddings, dim=0)
    tokens_all = torch.cat(sorted_values_tokens, dim=0)
    weights_all = torch.cat(sorted_values_weights, dim=0)

    print(embeddings_all.shape)
    print(tokens_all.shape)
    print(weights_all.shape)

    embeddings_all = embeddings_all.cpu().numpy()
    tokens_all = tokens_all.cpu().numpy()
    weights_all = weights_all.cpu().numpy()
    import numpy as np
    args_path = f"{root}/{pre_trained_model_name}/args.json"
    np.save(f"{root}/{pre_trained_model_name}/embeddings_all.npy", embeddings_all)
    np.save(f"{root}/{pre_trained_model_name}/tokens_all.npy", tokens_all)
    np.save(f"{root}/{pre_trained_model_name}/weights_all.npy", weights_all)

    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    root = 'pre_trained_model/'
    pre_trained_model = 'model_name' ##please put the model name here
    main(root, pre_trained_model)