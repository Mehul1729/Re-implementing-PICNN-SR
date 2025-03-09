# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
import tikzplotlib
from sklearn.metrics import mean_squared_error as calMSE
import sys

# for reading OF generated mesh data:
from readVesselMesh import read_vessel_mesh,\
						   proj_solu_from_ofmesh_2_mymesh
#VaryGeoDataset is a custom function that converts mesh data into array like trainable data to CNN in pytorch setting.

from dataset import VaryGeoDataset
from pyMesh import visualize2D, setAxisLabel,to4DTensor
from model import USCNNSep
from readOF import convertOFMeshToImage_StructuredMesh

nEpochs = int(sys.argv[2])

h=0.01
Itol=0


#Prepare data:

# --------------------------------------------------------------------------
# CodeBlock1: Converts OpenFoam data (both LF and HF): velocity (U), pressure (p), and temperature (T) fields into structured image format:

# convert high fidelity mesh data to image like data:
# mesh_hf : mesh grid
# nx_hf, ny_hf : number of points in x and y axes in the image data
# read_vessel_mesh: fucntion from read_vessel+Mesh folder

# reading the fine mesh OF data:
mesh_hf,nx_hf,ny_hf=read_vessel_mesh('ns_eqn_parabolic/3200/C') # mesh_hf is just a hcubemesh object

# converts fine-resolution velocity (u) and pressure (p) into image-like array:
# C-file has mesh connectivities, [0,1,0,1] are the boundaries
# 0.0 represents no padding
solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/U',
	                                              'ns_eqn_parabolic/3200/p'],
	                                            [0,1,0,1],0.0,False)
# this generantes an array data of shape (77, 49, 5)
# print(solu_hf1.shape)





# temperature mesh to image array: (not included in PDE though)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/T'],
	                                            [0,1,0,1],0.0,False)

# COncatenating Temperature with velocity and pressure data:
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:]],axis=2)

# solu_hf just has an extra dimension for temperature when compared to solu_hf1
#print(solu_hf.shape)


# Low Resolution data reading:
mesh_lf,nx_lf,ny_lf=read_vessel_mesh('Coarse/3200/C')
# conversion of low resolu mesh data to image array:
solu_lf=convertOFMeshToImage_StructuredMesh(nx_lf,ny_lf,'Coarse/3200/C',
	                                           ['Coarse/3200/U',
	                                            'Coarse/3200/p'],
												[0,1,0,1],1,False)

# Projecting open foam LR data onto stuctured grid:
[OFU_lf,OFV_lf,OFP_lf]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_lf)
# this is a very los resolution data of shape (14,9)  => very less number of pixels
#print(OFU_lf.shape)


# Adding noise to LR data (@@@ we can try the results with various noise levels)

# trying to add relative noise as a command line argument:
import os
import sys
relativeNoise = float(sys.argv[1])  # Read value from command-line argument
results_dir = f"Results_Noise_{relativeNoise}"
os.makedirs(results_dir, exist_ok=True)

# relativeNoise=0.05
#np.random.uniform(low=0.0, high=1.0, size=1)
inputRand=relativeNoise*np.random.randn(1, 2, 14, 9)
inputRand=torch.from_numpy(inputRand)
inputRand=inputRand.float().to('cuda')

# Converting low resolution velocity and pressure to torch tensors:
# reshape(1 batch, 1 channel, height, width):

u_lf=torch.from_numpy(OFU_lf).float().to('cuda').reshape([1,1,OFU_lf.shape[0],OFU_lf.shape[1]])
v_lf=torch.from_numpy(OFV_lf).float().to('cuda').reshape([1,1,OFV_lf.shape[0],OFV_lf.shape[1]])
p_lf=torch.from_numpy(OFP_lf).float().to('cuda').reshape([1,1,OFP_lf.shape[0],OFP_lf.shape[1]])


# Inlet velocity profile: (for boundary loss)
velofunc=lambda argc: 100*argc*(0.2-argc)  # this is pre-known boundary condition for inlet velocity => inlet velocity profile (along y-direc) varies with x-coordinate axis parabolically.
#pdb.set_trace()       # debugger

infertruth=torch.from_numpy(velofunc(mesh_hf.x[0,:])).float().to('cuda')
infertruth=infertruth[1:-1].reshape([1,47]) # removes boundary inlet velocity (x= 0 and x = 0.2) and reshapaes into 1*47 matrix
# infertruth will be used to track the deviations of CNN output from inlet velocity profile


# ??? defining and converting the high fidelity mesh again?
solu_hf1=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/U',
	                                              'ns_eqn_parabolic/3200/p'],
	                                            [0,1,0,1],0.0,False)
solu_hf2=convertOFMeshToImage_StructuredMesh(nx_hf,ny_hf,'ns_eqn_parabolic/3200/C',
	                                             ['ns_eqn_parabolic/3200/T'],
	                                            [0,1,0,1],0.0,False)
solu_hf=np.concatenate([solu_hf1,solu_hf2[:,:,2:]],axis=2)




HFMeshList=[mesh_hf]
train_set=VaryGeoDataset(HFMeshList)

#Set up paramters for training:
batchSize = 1        # Number of samples per training batch
NvarInput = 2        # Number of input variables (e.g., velocity components U, V)
NvarOutput = 1       # Number of output variables (e.g., super-resolved velocity field)

# Number of epochs are asked as user input in command line.
#Epochs = nEpoch# Read nEpochs from command-line
lr = 0.001           # Learning rate for optimizer (how much weights are updated per step)
Ns = 1               # Not explicitly used in this snippet, could refer to number of samples
nu = 0.01            # Viscosity parameter (used in Navier-Stokes equations)
criterion = nn.MSELoss()  # Mean Squared Error (MSE) loss function

# Padding Setup:
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)

#Set up model
model=USCNNSep(h,nx_hf,ny_hf,NvarInput,NvarOutput,'ortho').to('cuda')
model=model.to('cuda')
optimizer = optim.Adam(model.parameters(),lr=lr)

# Data loader just shuffles and creates batches for easy and effectiev training
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)



# TEST Ground Truth Data ---> Project OFSolution to MyMesh
[OFU_sb,OFV_sb,OFP_sb,OFT_sb]=proj_solu_from_ofmesh_2_mymesh(solu_hf,mesh_hf)
# OFU_sb has shape (77, 49)


#[BICUBICU,BICUBICV,_]=proj_solu_from_ofmesh_2_mymesh(solu_lf,mesh_hf)

T_hardimpo=torch.from_numpy(OFT_sb).reshape([1,1,OFT_sb.shape[0],OFT_sb.shape[1]]).float().to('cuda')
OFV_sb_cuda=torch.from_numpy(OFV_sb).reshape([1,1,OFV_sb.shape[0],OFV_sb.shape[1]]).float().to('cuda')




# Randomly Sampling Sparse HF data from the Open Foam test data:
# Sparse HF observations will be used for evaluation
from random import sample
nobs=200
idxpool=[i for i in range(77*49) if i>147]  # avoiding boundary points to avoid sampling near inlet and outlet
#idxsample=sample(idxpool,nobs)
idxsample=[i for i in idxpool if (i in range(49*10+2,49*11-2)) or (i in range(49*30+2,49*31-2)) or (i in range(49*50+2,49*51-2)) or (i in range(49*65+2,49*66-2))]
idx1=[i//49 for i in idxsample] #77
idx2=[i%49 for i in idxsample] #49
idx1=np.asarray(idx1)
# saving sparse HF data for later use 
np.savetxt('idx1.txt',idx1)
idx2=np.asarray(idx2)
np.savetxt('idx2.txt',idx2)



# --------> Transformations using Kernels:

#Define the geometry transformation for feeding into  CNN (Observation LR data may have non-rectangular shapes)

# Also these transformations are used for claculating differential terms in the Naviar Stokes Residual:

def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)

	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx

def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)

	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy

def psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(gt)  # Use max value in ground truth as reference
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)



# Define the model to train ---------------------------- :


# Tracks losses and errors for x-momentum, y-momentum, mass conservation, transport equation, and velocity/pressure errors:

def train(epoch):
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	TRes=0
	eU=0
	eV=0
	eP=0




	# Using the HF mesh info for loss calculation:
	for iteration, batch in enumerate(training_data_loader):
		[_,cord,_,_,_,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
		optimizer.zero_grad()  # Extracts mesh coordinates, Jacobian matrix (Jinv), and transformation coefficients (dxdxi, dydxi, dxdeta, dydeta



		# Preparing input data for CNN: (only u_lf and v_lf are used for training => only LR data)
		input=torch.cat([u_lf,v_lf],axis=1)
		input=input*(1+inputRand) # adding noise to input data



	# Saves initial input velocity fields (U, V) in text files for debugging:
		if epoch==1:
			np.savetxt(f'{results_dir}/InputU.txt',input[0,0,:,:].detach().cpu().numpy())
			np.savetxt(f'{results_dir}/InputV.txt',input[0,1,:,:].detach().cpu().numpy())



# Bicubic Interpolation function US is defined in the Model.py file along with the USCNN main model. It uses Upsample method of nn-class in Pytorch:

# Slicing is done on the following: [batch_size, num_channels, height, width]
		BICUBICU=model.US(input)[0,0,:,:].detach().cpu().numpy() # detach is used because this part is not being used for trainign. Bicubic thing is only for visualisation and comparision
		BICUBICV=model.US(input)[0,1,:,:].detach().cpu().numpy()

		output=model(input) # forward pass
		output_pad=udfpad(output) # padding the output (stride = 1)


# Initializing empty tensors for storing velocity corrections at observation points and sparse velocity observations from openfoam data (OFV_sb_cuda):
		VTEMP=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')
		VSparseOBS=torch.zeros([1,1,ny_hf,nx_hf]).float().to('cuda')

	# Cnn model output reshaping:
		outputV_tmep=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])

		# here : Correction of Output velcoity field using Sparse Observations:
		# idx1 and idx2 are for iterating for sparse observation points
		for ii in range(len(idx1)):
			# VTEMP stores the velociy difference between CNN O/P and Sparse Open Foam HF data
			VTEMP[0,0,idx1[ii],idx2[ii]]=OFV_sb_cuda[0,0,idx1[ii],idx2[ii]]-outputV_tmep[0,0,idx1[ii],idx2[ii]]

# VSparseOBS stores the randomly sampled sparse observations from openfoam:
			VSparseOBS[0,0,idx1[ii],idx2[ii]]=OFV_sb_cuda[0,0,idx1[ii],idx2[ii]]


		# Extracts U velocity from CNN output:
		outputU=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
   # Integrates the velocity differences in predicted V from sparse observations from OpenFoam data:
		outputV=VTEMP+output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])

	# Extracts P from CNN output:
		outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])

# Initializing zero tensors for storing PDE residuals:
		XR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda') #X-momnetum residual
		YR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda') # Y-momentum residual
		MR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda') # Mass continuity residual
		TR=torch.zeros([batchSize,1,ny_hf,nx_hf]).to('cuda') # Scalar transport residual


		for j in range(batchSize):
			#Impose BC:
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,:].reshape(1,nx_hf-2*padSingleSide) # up outlet bc zero gradient
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0  # down inlet bc
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 # right wall bc
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 # left  wall bc

			#outputU[j,0,0,0]=0.5*(outputU[j,0,0,1]+outputU[j,0,1,0])
			#outputU[j,0,0,-1]=0.5*(outputU[j,0,0,-2]+outputU[j,0,1,-1])
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])


			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx_hf-2*padSingleSide) # up outlet bc zero gradient

			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=torch.abs(model.source)			# down inlet bc


			# This code computes the error in the inferred inlet velocity profile compared to the ground truth:  (RMSE)
			einfer=torch.sqrt(criterion(infertruth,torch.abs(model.source))/criterion(torch.abs(model.source),torch.abs(model.source)*0))
			#print('Infer velocity====================',model.source[0])

			try:
				print('>>>>>>>Infer error<<<<<<< =======================',einfer.item())
				print('>>>>>>>model source<<<<<<< =======================',model.source)
			except:
				pass
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 					    # right wall bc
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 					    # left  wall bc

			#outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			#outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])



			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 								  # up outlet zero pressure
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx_hf-2*padSingleSide)      # down inlet zero gradient bc

			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny_hf-2*padSingleSide,1)    # right wall zero gradient bc
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny_hf-2*padSingleSide,1)     # left  wall zero gradient bc

			#outputP[j,0,0,0]=0.5*(outputP[j,0,0,1]+outputP[j,0,1,0])
			#outputP[j,0,0,-1]=0.5*(outputP[j,0,0,-2]+outputP[j,0,1,-1])
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])


# defining derivatives for PDE residuals:
		dudx=dfdx(outputU,dydeta,dydxi,Jinv)
		d2udx2=dfdx(dudx,dydeta,dydxi,Jinv)


		dudy=dfdy(outputU,dxdxi,dxdeta,Jinv)
		d2udy2=dfdy(dudy,dxdxi,dxdeta,Jinv)




		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)

		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)

		dpdx=dfdx(outputP,dydeta,dydxi,Jinv)
		d2pdx2=dfdx(dpdx,dydeta,dydxi,Jinv)

		dpdy=dfdy(outputP,dxdxi,dxdeta,Jinv)
		d2pdy2=dfdy(dpdy,dxdxi,dxdeta,Jinv)


		#scalar transport of temperature field:
		dTdx=dfdx(T_hardimpo,dydeta,dydxi,Jinv)
		d2Tdx2=dfdx(dTdx,dydeta,dydxi,Jinv)
		dTdy=dfdy(T_hardimpo,dxdxi,dxdeta,Jinv)
		d2Tdy2=dfdy(dTdy,dxdxi,dxdeta,Jinv)



		#Calculate PDE Residuals -----------------------------:::::



		continuity=(dudx+dvdy);  # incompressibility condition
		#continuity=-(d2pdx2+d2pdy2)-d2udx2-d2vdy2-2*dudy*dvdx
		#u*dudx+v*dudy
		momentumX=outputU*dudx+outputV*dudy # advection forces in x-direction

		# which describe how the velocity is transported by the flow:
		#-dpdx+nu*lap(u)
		forceX=-dpdx+nu*(d2udx2+d2udy2) # ( pressure gradient + viscous diffusion)



		# Xresidual:
		# This checks whether the left-hand side (convective forces) matches the right-hand side (pressure + viscous forces).
    # If Xresidual is nonzero, it means the CNN’s output does not fully satisfy the governing equation.
		Xresidual=(momentumX-forceX)


		#u*dvdx+v*dvdy
		momentumY=outputU*dvdx+outputV*dvdy # momentum transport in x - direction
		#-dpdy+nu*lap(v)
		forceY=-dpdy+nu*(d2vdx2+d2vdy2)
		Yresidual=(momentumY-forceY) # same force balancing residual but in y-direction


# HEat transport residual for temperature:
		#T*dudx + u*dTdx
		convecX=T_hardimpo*dudx+outputU*dTdx
		#T*dvdy + v*dTdy
		convecY=T_hardimpo*dvdy+outputV*dTdy
		#nu*laplacian
		diffusionT=nu*d2Tdy2+nu*d2Tdx2
		Tresidual=(convecY+convecX-diffusionT)*0







# Adding up all the residuals to get the losses
		loss=(criterion(Xresidual,Xresidual*0)+\
		  criterion(Yresidual,Yresidual*0)+\
		  criterion(continuity,continuity*0)+\
		  criterion(Tresidual,Tresidual*0))
		loss.backward()
		optimizer.step()



		# To store the loss values and analyze them separately:

		# Zeroes in the loss terms act as exlplicit target for criterion function so that it can continuoisly measure how far the loss is from the target:
		loss_xm=criterion(Xresidual, Xresidual*0)
		loss_ym=criterion(Yresidual, Yresidual*0)
		loss_mass=criterion(continuity, continuity*0)
		loss_T=criterion(Tresidual, Tresidual*0)


    # accumulating all the PDE residuals for each of the batches:
		xRes+=loss_xm.item()
		yRes+=loss_ym.item()
		mRes+=loss_mass.item()
		TRes+=loss_T.item()

		# [batch_size, num_channels, height, width]
		# Numpy arrauys of cnn outputs:
		CNNUNumpy=outputU[0,0,:,:].cpu().detach().numpy()
		CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
		CNNPNumpy=outputP[0,0,:,:].cpu().detach().numpy()

		psnr_U = psnr(OFU_sb, CNNUNumpy)
		psnr_V = psnr(OFV_sb, CNNVNumpy)
        
		print(f"PSNR for U: {psnr_U:.2f} dB")
		print(f"PSNR for V: {psnr_V:.2f} dB")
        
		np.savetxt(f"{results_dir}/PSNR_U.txt", [psnr_U])
		np.savetxt(f"{results_dir}/PSNR_V.txt", [psnr_V])



    # accumulating ground truth errors: (normalised as well)
		eU=eU+np.sqrt(calMSE(OFU_sb,CNNUNumpy)/calMSE(OFU_sb,OFU_sb*0))
		eV=eV+np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))
		eP=eP+np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0))

	# error in total velcity magnitude:  (sqrt(u^2 + v^2))
		eVmag=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(CNNUNumpy**2+CNNVNumpy**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))


	# comparision of CNN results with BIcubic Interpolation results:
	# CalMSE for calculating the the mean squared loss:
		eVBICUIC=np.sqrt(calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2),np.sqrt(BICUBICU**2+BICUBICV**2))/calMSE(np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2),np.sqrt(OFU_sb[1:-1,1:-1]**2+OFV_sb[1:-1,1:-1]**2)*0))



		# printing the error values per batch:
		print('VelMagError_CNN=',eVmag)
		print('VelMagError_BI=',eVBICUIC)
		print('P_err_CNN=',np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0)))



	# Epoch wise loss (Mean) printing:
	# PDE resiudals:
	print('Epoch is ',epoch)
	print("xRes Loss is", (xRes/len(training_data_loader))) # PDE Momentum res.(x)
	print("yRes Loss is", (yRes/len(training_data_loader))) # PDE momentum res.(y)
	print("mRes Loss is", (mRes/len(training_data_loader))) # PDE mass conserv.res
	print("TRes Loss is", (TRes/len(training_data_loader)))

 	# Ground truth errors:
	print("eU Loss is", (eU/len(training_data_loader))) # x-velocity
	print("eV Loss is", (eV/len(training_data_loader))) # y-velocity
	print("eP Loss is", (eP/len(training_data_loader))) # Pressure
	#TOL=[0.07,0.06,0.05,0.04,0.03,0.02,0.01]

	# saving Bicubic interpolation after frst epoch (for baseline comparision with cnn):
	if epoch==1:
		np.savetxt(f'{results_dir}/BIErrorVmag.txt',eVBICUIC*np.ones([4,4]))

	# saving model checkpoints
	# Saves the model (.pth file) under these conditions:
	if (einfer.item()<0.05) or epoch%nEpochs==0 or epoch%1000==0 or epoch ==100:
		model_path = f"{results_dir}/model_epoch_{epoch}.pth"  # Save inside results directory
		torch.save(model, model_path)


	# plotting the cnn predicted velocity field:
		fig0=plt.figure()
		ax=plt.subplot(2,3,2)
		_,cbar=visualize2D(ax,mesh_hf.x,
			                  mesh_hf.y,
			           np.sqrt(outputU[0,0,:,:].cpu().detach().numpy()**2+\
			           		   outputV[0,0,:,:].cpu().detach().numpy()**2),'vertical',[0,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('CNN')
		cbar.set_ticks([0,0.25,0.5,0.75,1.0]) # ticks for x-axis
		#ax.set_aspect('equal')

# plottig the truth velocity field
		ax=plt.subplot(2,3,3)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			           np.sqrt(OFU_sb**2+\
			           		   OFV_sb**2),'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Truth')
		#ax.set_aspect('equal')

#
		ax=plt.subplot(2,3,4)
		_,cbar=visualize2D(ax,mesh_hf.x,
			           		  mesh_hf.y,
			                  2*VSparseOBS[0,0,:,:].cpu().detach().numpy(),'vertical',[0,1.0]) # multiplying by 2 for better contrast
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Sparse Observation')
		#ax.set_aspect('equal')

# plotting intput Low resolution data:
		ax=plt.subplot(2,3,1)
		_,cbar=visualize2D(ax,mesh_lf.x,
			           		  mesh_lf.y,
			           np.sqrt(input[0,0,:,:].cpu().detach().numpy()**2+\
			           		   input[0,1,:,:].cpu().detach().numpy()**2**2),'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Input LR data')
		#ax.set_aspect('equal')


# plotting Bicubic interpolation prdicitions
		ax=plt.subplot(2,3,5)
		#pdb.set_trace()
		_,cbar=visualize2D(ax,mesh_hf.x[1:-1,1:-1],
			           		  mesh_hf.y[1:-1,1:-1],
			           		  np.sqrt(BICUBICU**2+BICUBICV**2),
							  'vertical',[0,1.0])
		cbar.set_ticks([0,0.25,0.5,0.75,1.0])
		setAxisLabel(ax,'p')
		ax.set_title('Bicubic Interpolation')
		#ax.set_aspect('equal')


		'''
		ax=plt.subplot(2,3,4)
		visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           outputP[0,0,:,:].cpu().detach().numpy(),'vertical',[0,0.35])
		setAxisLabel(ax,'p')
		ax.set_title('Super-resolved '+'Pressure')

		ax=plt.subplot(2,3,5)
		visualize2D(ax,mesh_hf.x,
			           mesh_hf.y,
			           OFP_sb[:,:],'vertical',[0,0.35])
		setAxisLabel(ax,'p')
		ax.set_title('True '+'Pressure')
		'''


# compares inlet velocity profile :
		ax_=plt.subplot(2,3,6)

		ax_.plot(mesh_hf.x[0,1:-1].reshape([1,47]),model.source.cpu().detach().numpy(),'x',label='Inferred',color='blue')
		ax_.plot(mesh_hf.x[0,:],velofunc(mesh_hf.x[0,:]),'--',label='True')
		setAxisLabel(ax_,'p')
		ax_.set_ylabel(r'$v$')
		ax_.set_title('Inlet '+'Velocity Profile')


# saving all the plots in a pdf file
		fig0.tight_layout(pad=1)
		fig0.savefig(f"{results_dir}/Epoch_{epoch}_Transport.pdf", bbox_inches='tight')
		plt.close(fig0)




	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   (mRes/len(training_data_loader)), (TRes/len(training_data_loader)) ,(eU/len(training_data_loader)),\
		   (eV/len(training_data_loader)), (eP/len(training_data_loader)),model.source.detach().cpu().numpy(),einfer.item()








# Saving all the residuals and losses per epoch in lists separately:
# Also, saving the time spent after each epoch:

XRes=[];YRes=[];MRes=[];CRes=[]
EU=[];EV=[];EP=[]
Iinlet=[]
TotalstartTime=time.time()
EINFER=[]
for epoch in range(1,nEpochs+1):
	tic=time.time()
	xres,yres,mres,cres,eu,ev,ep,infer,einferr=train(epoch)
	print('Time of this epoch=',time.time()-tic)
	EINFER.append(einferr)
	XRes.append(xres)
	YRes.append(yres)
	MRes.append(mres)
	CRes.append(cres)
	EU.append(eu)
	EV.append(ev)
	EP.append(ep)
	Iinlet.append(infer)
	if einferr<0.01:
		break
TimeSpent=time.time()-TotalstartTime



# Plotting graphs for residuals:
plt.figure()
plt.plot(XRes,'-o',label='X-momentum Residual')
plt.plot(YRes,'-x',label='Y-momentum Residual')
plt.plot(MRes,'-*',label='Continuity Residual')
plt.plot(CRes,'-.',label='Transport Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig(f'{results_dir}/convergence.pdf',bbox_inches='tight')
tikzplotlib.save(f'{results_dir}/convergence.tikz')

plt.figure()
plt.plot(EU,'-o',label=r'$u$')
plt.plot(EV,'-x',label=r'$v$')
plt.plot(EP,'-*',label=r'$p$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig(f'{results_dir}/error.pdf',bbox_inches='tight')
tikzplotlib.save(f'{results_dir}/error.tikz')
EU=np.asarray(EU)
EV=np.asarray(EV)
EP=np.asarray(EP)
XRes=np.asarray(XRes)
YRes=np.asarray(YRes)
MRes=np.asarray(MRes)
CRes=np.asarray(CRes)
Iinlet=np.asarray(Iinlet)
np.savetxt(f"{results_dir}/EU.txt", EU)
np.savetxt(f"{results_dir}/EV.txt", EV)
np.savetxt(f"{results_dir}/EP.txt", EP)
np.savetxt(f"{results_dir}/Iinlet.txt", Iinlet.squeeze())
np.savetxt(f"{results_dir}/XRes.txt", XRes)
np.savetxt(f"{results_dir}/YRes.txt", YRes)
np.savetxt(f"{results_dir}/MRes.txt", MRes)
np.savetxt(f"{results_dir}/CRes.txt", CRes)
np.savetxt(f"{results_dir}/TimeSpent.txt", np.zeros([2,2]) + TimeSpent)
np.savetxt(f"{results_dir}/EINFER.txt", np.asarray(EINFER))




















"""
































'''
			dudx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputU[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputU[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2udx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dudx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dudx)*dydxi[j:j+1,0:1,2:-2,2:-2])
			dvdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputV[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputV[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2vdx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dvdx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dvdx)*dydxi[j:j+1,0:1,2:-2,2:-2])

			dudy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputU[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputU[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2udy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dudy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dudy)*dxdeta[j:j+1,0:1,2:-2,2:-2])
			dvdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputV[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputV[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2vdy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dvdy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dvdy)*dxdeta[j:j+1,0:1,2:-2,2:-2])

			dpdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputP[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputP[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			dpdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputP[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputP[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])

			continuity=dudx[:,:,2:-2,2:-2]+dudy[:,:,2:-2,2:-2];
			#u*dudx+v*dudy
			momentumX=outputU[j:j+1,:,2:-2,2:-2]*dudx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdx
			#-dpdx+nu*lap(u)
			forceX=-dpdx[0:,0:,2:-2,2:-2]+nu*(d2udx2+d2udy2)
			# Xresidual
			Xresidual=momentumX[0:,0:,2:-2,2:-2]-forceX

			#u*dvdx+v*dvdy
			momentumY=outputU[j:j+1,:,2:-2,2:-2]*dvdx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdy
			#-dpdy+nu*lap(v)
			forceY=-dpdy[0:,0:,2:-2,2:-2]+nu*(d2vdx2+d2vdy2)
			# Yresidual
			Yresidual=momentumY[0:,0:,2:-2,2:-2]-forceY
			'''

"""



