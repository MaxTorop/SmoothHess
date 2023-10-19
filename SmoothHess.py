import torch   
import numpy as np 
from Datasets import *
# from tqdm.auto import tqdm 
from tqdm import tqdm 

############ Inputs #############
# (1) model: NN of interest
# (2) logit_class: int, logit class indicating function R^d --> R of interest in NN. In theory could correspond to any neuron 
# (3) iterations: int, number of iterations to MC estimate
# (4) x: tensor, in its original shape, point of interestp
# (4) reflect_samples: bool,  flip each sample from MVN along origin 
# (5) TODO: std: scalar, replace with cov 
# (6) n1: int, number of samples to use per iteration. if reflect_samples == True then n1 * 2 samples used per iteration
# (7) SG: bool, estimate SmoothGrad
# (8) SH: bool, estimate SmoothHess
# (9) savedir_stein,
# (10) savename_stein,  
########### Outputs ###############
# (1) SmoothGrad 
# (2) SmoothHess
def SmoothGeneral(model, logit_class, iterations, x, ReflectSamples, std, n1, SG = True, SH = True, binary = False, Symmetrize = False, function = "logit"): #, OriginalDim = (3, 32, 32)):
    assert SG == True or SH == True, "Neither SmoothGrad or SmoothHess is being computed"
    assert function == "Logit" or function == "SoftMax" or function == "Loss" or "Penult" in function, "Function must be logit SoftMax or Loss"

    OriginalDim = x.shape 
    x = x.flatten() 
    input_dim = len(x) 
    SoftMax = torch.nn.Softmax(dim = 1)

    # Iteratively Update With Sum Of Rank 1 Matrices 
    SmoothHess = torch.zeros((len(x), len(x)))
    SmoothGrad = torch.zeros((len(x))) 

    # If cov is an array containing 1 element then scalar

    # Else 
    mean            = x #
    sig             = std #.1 #0.01
    cov_scalar      = sig 
    chol_scalar     = torch.sqrt(torch.tensor(sig).float())
    cov_chol_scalar = 1 / cov_scalar * chol_scalar 

    if "Penult" in function:
        neuron_ind = int(function.split("_")[1])

    # for i in tqdm(range(iterations), position = 0, leave = True ):
    # for i in tqdm(range(iterations)):
    for i in range(iterations): 
        # if i % 5 == 0 :
        #     print(i)         
        MVN_Unit_Samples = torch.normal(mean = torch.zeros(input_dim * n1), std = torch.ones(input_dim * n1))
        MVN_Unit_Samples = MVN_Unit_Samples.view(n1, input_dim)
     
        # Reflects samples along origin 
        if ReflectSamples:
            MVN_Unit_Samples = torch.concat(( MVN_Unit_Samples, MVN_Unit_Samples * -1), dim = 0)
    
        ### Unit normal samples --> Normal of interest samples 
        Samples = mean.unsqueeze(dim =1).cuda() + chol_scalar *  MVN_Unit_Samples.T.float().cuda()
        
        ##### Get Gradients - Computational Backbone 
        Samples = Samples.cuda()
        Samples.requires_grad = True
        
        if not binary: 
            if function == "Logit":
                pred = model(Samples.T.view(-1, *OriginalDim))[:,logit_class].sum()
            elif function == "SoftMax":
                pred = SoftMax(model(Samples.T.view(-1, *OriginalDim)))[:,logit_class].sum()
            elif "Penult" in function:
                pred = model(Samples.T.view(-1, *OriginalDim), penult = neuron_ind).sum()
        elif binary: 
            pred = model(Samples.T.view(-1, *OriginalDim)).sum()

        pred.backward()
        Grads = Samples.grad.T.clone() 
        
        if SH == True: 
            # Sum Of Rank 1 Matricies In Expectation  
            Sum_Rank_1 = torch.matmul(MVN_Unit_Samples.T.float().cuda(),Grads) 
            Sum_Rank_1 = Sum_Rank_1 / len(Samples.T)

            output_SH   = cov_chol_scalar * Sum_Rank_1.detach().cpu()   
            SmoothHess += output_SH 

        if SG == True: 
            output_SG   = Grads.T.mean(dim = 1).detach().cpu() 
            SmoothGrad += output_SG 

        # print("...")
        # print(i)
        # print((SmoothGrad / (i+1)).detach().cpu().norm()) 
        # print((SmoothHess / (i+1)).detach().cpu().norm()) 

    SmoothHess = SmoothHess / (i + 1) 
    SmoothGrad = SmoothGrad / (i + 1)

    # print((SmoothHess/ (i+1) - SmoothHess.T / (i+1)).norm())

    if Symmetrize:
        SmoothHess = 1/2 * (SmoothHess + SmoothHess.T)

    return SmoothGrad , SmoothHess 

