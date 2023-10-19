import torch 
from torch.autograd.functional import hessian 
from tqdm.auto import tqdm 
import copy 

# Local imports 
from SmoothHess import SmoothGeneral 

def Generate_Grad_Hess_From_List(model, data, args, method = "Smooth", savedir = None): 

    OriginalDim = args['OriginalDim'] 
    binary = args['binary']
    function = args['function_use']

    # Store Computed Gradients and Hessians 
    GradList = torch.zeros(len(data), len(data[0].flatten()))
    HessList = torch.zeros(len(data), len(data[0].flatten()), len(data[0].flatten())) 

    if "Penult" in function and method == "SoftPlus" and args['dataset'] != "CIFAR10": # modify network once in the beginning for penult non-cifar10 
        # old_model = copy.deepcopy(model)

        model.f = model.f[:-2]
        neuron_num = int(function.split("_")[1])
        selection_layer = torch.nn.Linear(in_features = model.f[-1].out_features, out_features = 1, bias = False).cuda() 
        selection_layer.weight.data = torch.zeros(*selection_layer.weight.data.shape).float() 
        selection_layer.weight.data[0,neuron_num] = torch.tensor(1).float() 
        selection_layer.requires_grad = True 
        selection_layer = selection_layer.cuda()
        model.f.append(selection_layer)

    for ind in range(len(data)): # compute gradient and hessian for each data point 

        if ind % 5 == 0:
            print("Computing for point " + str(ind)) 

        point = data[ind,:].cuda()
        model_output =  model(point.view(1, *OriginalDim))
        logit_class= model_output.argmax().detach().cpu().numpy() # Predicted class to be used 

        if method == "SoftPlus": # Compute SoftPlus Hessian and Gradient
            Grad, Hess = GetSPGradHess(model, point, logit_class, function = args['function_use'], args = args)
            Grad, Hess = torch.tensor(Grad), torch.tensor(Hess)

        elif method == "Smooth": # Compute SmoothHess and SmoothGrad 
            Grad, Hess = SmoothGeneral(model, logit_class, iterations = args['iterations'], x = point, ReflectSamples = args['ReflectSamples'], std = args['sigma'], n1 = args['n1'], SG = True, SH = True, binary= binary, Symmetrize = args['Symmetrize'], function = args['function_use']) #, savedir = args['savedir_stein'])

        elif method == "Vanilla":
            if function != "SoftMax":
                point = point.view(OriginalDim)
                point.requires_grad = True 

                if "Penult" in function:
                    penult = int(function.split("_")[1])

                    preds               = model(point.view(1, *OriginalDim), penult = penult).sum()
                    preds.backward()
                    Grad = point.grad.T.clone().flatten()  
                elif function == "Logit":
                    prediction_index          = int(model(point.view(1, *OriginalDim)).argmax())

                    preds                     = model(point.view(1, *OriginalDim))[:,prediction_index].sum()
                    preds.backward()
                    Grad = point.grad.T.clone().flatten()  

                Hess = torch.zeros((len(Grad), len(Grad))).cuda()    

            elif function == "SoftMax": #ToDo Clean 
                Grad, Hess = GetVanillaGradHess(model, point, logit_class, function, args)

        GradList[ind, :] = Grad.detach().cpu()
        HessList[ind,:,:] = Hess.detach().cpu()

    # unmidofy network 
    # if "Penult" in function and method == "SoftPlus" and args['dataset'] != "CIFAR10": 
    #     model = old_model 
    return GradList, HessList 


def GetVanillaGradHess(model, point, logit_class, function, args):
    assert function == "SoftMax", "Non-SoftMax Vanilla Hess is 0, This Method Is Overkill Just Compute Vanilla Grad"
    #### If function = SoftMax take time to compute Hessian, else just compute grad and set Hessian to 0
    xc = point.clone() 
    xc = torch.nn.Parameter(xc)
    xc.requires_grad = True

    if not args['dataset'] == "CIFAR10":
        xc = xc.view(args['OriginalDim'])
        #### If using softmax 
        softmax = torch.nn.Softmax()
        selection_layer = torch.nn.Linear(in_features = args['num_classes'], out_features = 1, bias = False).cuda() 
        selection_layer.weight.data = torch.zeros(*selection_layer.weight.data.shape).float() 
        selection_layer.weight.data[0,logit_class] = torch.tensor(1).float() 
        selection_layer.requires_grad = True 

        if function == "SoftMax":
            model.f.append(softmax)
        model.f.append(selection_layer)

        model = model.cuda() 
        outputs = model(xc.cuda())
        grad = torch.autograd.grad(outputs, xc)
        hess = hessian(model, xc) # Change out to functorch version 

        hess = hess.detach().cpu() #.detach().cpu().numpy()
        grad = grad[0].detach().cpu() #.detach().cpu().numpy()

        if function == "SoftMax":
            model.f = model.f[:-2]
    else:
        xc = xc.view(args['OriginalDim']).unsqueeze(dim = 0)
        softmax = torch.nn.Softmax()
        selection_layer = torch.nn.Linear(in_features = args['num_classes'], out_features = 1, bias = False).cuda() 
        selection_layer.weight.data = torch.zeros(*selection_layer.weight.data.shape).float() 
        selection_layer.weight.data[0,logit_class] = torch.tensor(1).float() 
        selection_layer.requires_grad = True 

        if function == "SoftMax":
            model.softmax = softmax
        model.selection_layer = selection_layer
        model = model.cuda() 
        
        
        outputs = model(xc.cuda())
        grad = torch.autograd.grad(outputs, xc)
        grad = grad[0].detach().cpu() 

        hess = torch.zeros((len(point.flatten()), len(point.flatten()))).cuda() 

        for i in range(len(point.flatten())):
            if i % 100 == 0:
                print(i)
            v = torch.zeros(len(point.flatten())).cuda() 
            v[i] = 1 
            v = v.view(xc.shape)
            _, hvprod = torch.autograd.functional.hvp(model, xc, v)
            hess[i,:] = hvprod.flatten() 
        
        # reset so unused
        model.softmax = None
        model.selection_layer = None   


    hess = 1/2 *  ( hess + hess.T)

    return torch.tensor(grad).flatten(), torch.tensor(hess)




def GetSPGradHess(model, x, logit_class, function = "Logit", savedir= "./", savename = "a", beta = -1, num_classes = 10, args = None):
    
    xc = x.clone() 
    xc = torch.nn.Parameter(xc)
    xc.requires_grad = True
    beta = args['beta']
    model.replace_relu_softplus(args['beta']) #Note -- don't do this for Spiro 
    input_dim = len(xc.flatten())


    if args['dataset'] == "MNIST" or args['dataset'] == "FMNIST":

        if not "Penult" in function:
            xc = xc.view(args['OriginalDim'])
            #### If using softmax 
            softmax = torch.nn.Softmax()
            selection_layer = torch.nn.Linear(in_features = num_classes, out_features = 1, bias = False).cuda() 
            selection_layer.weight.data = torch.zeros(*selection_layer.weight.data.shape).float() 
            selection_layer.weight.data[0,logit_class] = torch.tensor(1).float() 
            selection_layer.requires_grad = True 

            if function == "SoftMax":
                model.f.append(softmax)
            model.f.append(selection_layer)

            model = model.cuda() 

            outputs = model(xc.cuda())
            grad = torch.autograd.grad(outputs, xc)
            hess = hessian(model, xc) 

            hess = hess.detach().cpu() 
            grad = grad[0].detach().cpu() 

            # Set Weight And Bias Back     
            if function == "SoftMax":
                model.f = model.f[:-2]
            elif function == "Logit":
                model.f = model.f[:-1]

        else:
            xc = xc.view(args['OriginalDim'])

            model.replace_relu_softplus(beta = beta ) 
            outputs = model(xc.cuda())
            grad = torch.autograd.grad(outputs, xc)
            hess = hessian(model, xc) 

            hess = hess.detach().cpu() 
            grad = grad[0].detach().cpu() 

    # CIFAR10 requires seperate method using Hessian Vector Products to Generate Hessian one column at a time 
    elif args['dataset'] == "CIFAR10":
        if not "Penult" in function:

            xc = xc.view(args['OriginalDim']).unsqueeze(dim = 0)
            softmax = torch.nn.Softmax()
            selection_layer = torch.nn.Linear(in_features = num_classes, out_features = 1, bias = False).cuda() 
            selection_layer.weight.data = torch.zeros(*selection_layer.weight.data.shape).float() 
            selection_layer.weight.data[0,logit_class] = torch.tensor(1).float() 
            selection_layer.requires_grad = True 

            if function == "SoftMax":
                model.softmax = softmax
            model.selection_layer = selection_layer
            model = model.cuda() 
            
            
            outputs = model(xc.cuda())
            grad = torch.autograd.grad(outputs, xc)
            grad = grad[0].detach().cpu() 

            hess = torch.zeros(input_dim, input_dim).cuda() 

            for i in range(input_dim):
                if i % 100 == 0:
                    print(i)
                v = torch.zeros(input_dim).cuda() 
                v[i] = 1 
                v = v.view(xc.shape)
                _, hvprod = torch.autograd.functional.hvp(model, xc, v)
                hess[i,:] = hvprod.flatten() 
            
            #### symmetrize
            hess = 1/2 *  ( hess + hess.T)
            hess = hess.detach().cpu()
            # reset so unused
            model.softmax = None
            model.selection_layer = None 

    
        else:
            xc = xc.view(args['OriginalDim']).unsqueeze(dim = 0)

            penult = int(function.split("_")[1])
            # Establish function if non-existant
            if model.penult_forward is None:
                model.penult_forward  = lambda x : model.f(x).flatten(start_dim = 1)[:, penult]
     
            # outputs = model.penult_forward_414(xc.cuda())
            outputs = model.penult_forward(xc.cuda())

            grad = torch.autograd.grad(outputs, xc)

            grad = grad[0].detach().cpu().flatten() 
            hess = torch.zeros((input_dim, input_dim)).cuda() 
          
            for i in tqdm(range(input_dim)):
                if i % 100 == 0:
                    print(i)
                v = torch.zeros(input_dim).cuda() 
                v[i] = 1 
                v = v.view(xc.shape)
                # _, hvprod = torch.autograd.functional.hvp(model.penult_forward_414, xc, v)
                _, hvprod = torch.autograd.functional.hvp(model.penult_forward, xc, v)

                hess[i,:] = hvprod.flatten() 
            
            #### Symmetrize 
            hess = 1/2 *  ( hess + hess.T)
            hess = hess.detach().cpu()

    model.replace_softplus_relu() 

    return torch.tensor(grad).flatten(), torch.tensor(hess) 


