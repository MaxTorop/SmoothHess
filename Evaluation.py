import torch
import numpy
import pandas as pd 
from tqdm.auto import tqdm 
from utils import * 

def Compute_PMSE(model, n1, n2, data, OriginalDim, function, rs, GradList, HessList):
    # assert (GradList is None and HessList is None and not loaddir is None) or (not GradList is None and not HessList is None and loaddir is None) 
    print("Computing P-MSE")    
    losses = {}         

    # If Penult: save penultimate neuron index 
    if "Penult" in function:
        penult = int(function.split("_")[1])

    for r in rs: 
        losses[r] = {"FirstOrder" : [], "SecondOrder" : [], "SecondBetterPct" : []}

        # for ind in tqdm(range(len(data)), position = 0, leave = True) :
        for ind in range(len(data)): 
            # print("Data Index "  + str(ind))

            if ind % 5 == 0:
                print(ind) 


            x = data[ind]
            x = x.cuda() 

            # Agglomerates P-MSE over batches 
            Loss_FirstOrder_Builder = 0 
            Loss_SecondOrder_Builder = 0
            
            for b_ind in range(n2): 
     
                Samples        = SampleFromBall(len(x.flatten()),n1,r)

                G = GradList[ind,:].cuda() 
                H = HessList[ind,:,:].cuda() 


                if "Penult" in function:
                    predictions               = model(x.view(1, *OriginalDim) + Samples.view((n1, *OriginalDim)), penult = penult)
                    x_prediction              = model(x.view(1, *OriginalDim), penult = penult).squeeze()

                elif function == "Logit":
                    prediction_index          = int(model(x.view(1, *OriginalDim)).argmax())
                    predictions               = model(x.view(1, *OriginalDim) + Samples.view((n1, *OriginalDim)))[:,prediction_index]
                    x_prediction              = model(x.view(1, *OriginalDim))[:, prediction_index]

                # elif function == "SoftMax":
                # prediction_index          = int(model(x.view(1, *OriginalDim)).argmax())
                #     predictions               = SoftMax(model(x.view(1, *OriginalDim) + Samples.view((n1, *OriginalDim))))[:,prediction_index]
                #     x_prediction              = SoftMax(model(x.view(1, *OriginalDim)))[:, prediction_index]
                        
                Taylor_Second_Order_Preds    = x_prediction + TaylorSecondOrder(G.flatten(), H, Samples, use_hess = True)
                Taylor_First_Order_Preds     =  x_prediction + TaylorSecondOrder(G.flatten(), H, Samples, use_hess = False)

                Loss_SecondOrder             = float((predictions - Taylor_Second_Order_Preds).square().mean().detach().cpu().numpy()) 
                Loss_FirstOrder              = float((predictions - Taylor_First_Order_Preds).square().mean().detach().cpu().numpy()) 

                Loss_FirstOrder_Builder += Loss_FirstOrder / n2
                Loss_SecondOrder_Builder += Loss_SecondOrder / n2 

            losses[r]['FirstOrder'].append(Loss_FirstOrder_Builder)
            losses[r]['SecondOrder'].append(Loss_SecondOrder_Builder)
            losses[r]['SecondBetterPct'].append(int(Loss_SecondOrder_Builder < Loss_FirstOrder_Builder))

        # Conver Lists To Tensors
        losses[r]['FirstOrder'] = torch.tensor(losses[r]['FirstOrder'])
        losses[r]['SecondOrder'] = torch.tensor(losses[r]['SecondOrder'])
        losses[r]['SecondBetterPct'] = torch.tensor(losses[r]['SecondBetterPct'])

    losses = pd.DataFrame.from_dict(losses)
    # losses.to_csv(Results_Save_Dir + "P_" + str(args['smooth_param'])  + ".csv")
    
    return losses  