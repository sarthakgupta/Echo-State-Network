#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:57:38 2018

@author: sarthak
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from time import time

np.random.seed(430)

#%%
###################################################################
####Creating Golden Output and Input
###################################################################
tmax=150
dt=0.08
div_unit=1/dt
num_points=div_unit*tmax

t_start=0

x0,y0=0,1
mu=2

scale=4

def lorenz(state, t, mu):
    x,y=state
    dxdt=y
    dydt=mu*(1-np.power(x,2))*y-x
    
    return dxdt,dydt

def sim_osc(state, t, mu):
    x,y=state
    dxdt=mu*y
    dydt=-(mu*x)
    
    return dxdt,dydt

###For freq controlled oscillator###
t_step=4
num_freq_steps=5
num_inputs=1000
train_split=0.80

t_end=t_start+(t_step*num_inputs)*train_split

t = np.linspace(0, t_step, div_unit*t_step)

for loop_input in range(0,num_inputs+int(t_start/t_step)):
    rand_in=np.random.randint(1,num_freq_steps+1)
#    rand_in=4
    f=odeint(sim_osc,(x0,y0),t, args=(rand_in,))
    state1,state2=f.T
    x0,y0=f[f.shape[0]-1,:]
    in_val=np.ones((f.shape[0],1))*rand_in
    f=np.column_stack((f,in_val))
    if(loop_input==0):
        f1=f
    else:
        f1=np.row_stack((f1,f)) 
            
golden_out=np.matrix(f1)/np.max(f1[:,:])
t = np.linspace(0, t_start+t_step*num_inputs, div_unit*(t_start+t_step*num_inputs))

###Ploting golden Data### 
fig = plt.figure(figsize=(10,4))
plt.subplot(3, 1, 1)
plt.plot(t,golden_out[:,0], 'b-', label='Golden Output (x1)', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Variable ::x1', fontsize=12)
plt.title('Golden Output vs. System Output', fontsize=16)
plt.legend(fontsize=12)

plt.subplot(3, 1, 2)    
plt.plot(t,golden_out[:,1], 'b-', label='Golden Output (x2)', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Variable ::x2', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(3, 1, 3)    
plt.plot(t,golden_out[:,2], 'b-', label='System Input', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Input::u1', fontsize=12)
plt.legend(fontsize=12)

#%%
###################################################################
#### Train_Test Data
###################################################################
#Using ODEint
traintest_cutoff = int(0.8*len(t))
frequency_control1=np.multiply(golden_out[:,2],1)
#frequency_control=np.column_stack((np.matrix(np.ones((len(t),1))),frequency_control1))
frequency_control=frequency_control1
frequency_output=np.multiply(golden_out[:,:2],2)+0.5

train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]

window_tr = range(int(len(train_output)/4),int(len(train_output)/4+2000))
plt.figure(figsize=(10,4))
plt.plot(train_ctrl[window_tr,0],label='control1')
#plt.plot(train_ctrl[window_tr,1],label='control2')
plt.plot(train_output[window_tr],label='target')
#plt.plot(pred_train[window_tr],label='model')
plt.legend(fontsize='x-small')
plt.title('training (excerpt)')
plt.ylim([-0.1,1.1])

#%%
###################################################################
#### Model Initilaizations
###################################################################
n_ip=1
n_rr=1000
n_op=2

#sparse_1::FF, sparse_2::Recurrent, sparse_3::Output Layer
sparse_1=1
sparse_2=0.25
sparse_3=1

w1_init_range=0.5
w2_rr_init_range=1
w2_bias_range=2
w3_init_range=0.8
w_fb_init_range=1
w_e_init_range=1

spectral_radius=0.25
noise_gain_train=0.001
noise_gain_test=0.0

#Fixed Feature Expansion weights, w1
w1=np.random.uniform(-w1_init_range,w1_init_range,(n_rr,n_ip))

#Making Sparse Connections in Recurrent Layer
w2_rr=np.zeros((n_rr,n_rr))
w2_rr_bar=np.zeros((n_rr,n_rr))

for loop_w2_rr in range(0,w2_rr.shape[0]):
    sparse_sel=sorted(np.random.choice(w2_rr.shape[1],int(sparse_2*w2_rr.shape[1]),replace=False))
    w2_rr[loop_w2_rr,sparse_sel]=np.random.uniform(-w2_rr_init_range,w2_rr_init_range,(int(sparse_2*w2_rr.shape[1])))

w2_rr_bar[w2_rr!=0]=1

w2_rr=np.matrix(np.random.uniform(-1,1,(n_rr,n_rr)))
w2_rr=np.multiply(spectral_radius/np.real(np.max(np.linalg.eigvals(w2_rr))), w2_rr)
print("Spectral_Radius w2_rr::",np.real(np.max(np.linalg.eigvals(w2_rr))))
        
w2_rr=np.multiply(w2_rr,w2_rr_bar)        
w2_rr_init=w2_rr

#Feedback Weights - Random and Fixed
w_fb=np.random.uniform(-w_e_init_range,w_e_init_range,(n_rr,n_op))

#%%
###################################################################
####Initialize state variables
###################################################################
Ih_tm1=np.zeros((n_rr,1))
Ie_tm1=np.zeros((n_op,1))
Io_tm1=np.zeros((n_op,1))
IW_tm1=np.zeros((n_rr,1))

Ih_t=np.zeros((n_rr,1))
Ie_t=np.zeros((n_op,1))
Io_t=np.zeros((n_op,1))
IW_t=np.zeros((n_rr,1))

eta=0.001
#%%
###################################################################
#### Model Implementation :: Offline Learning
###################################################################
st=np.zeros((n_op,1))
curr_time=time()

#ip_scale=np.matrix([0.01,3])
#ip_shift=np.matrix([0,0])
ip_scale=1
ip_shift=0

op_scale=np.matrix([1.12,1.12])
op_shift=np.matrix([-0.7,-0.7])

train_ctrl1=np.multiply(train_ctrl,ip_scale)+ip_shift

ip_data=np.matrix(train_ctrl1[:,:])
u_t_data=np.matmul(ip_data,w1.T)
train_output1=np.multiply(train_output,op_scale)+op_shift

x_train=np.matrix(np.zeros((len(train_ctrl),n_rr+n_ip)))
y_train=np.matrix(np.zeros((len(train_ctrl),n_op)))

#Calculating States
for i in range(len(train_ctrl)):
    u_t=u_t_data[i,:].T
    if(i>=1):
        st=np.matrix(train_output1[i-1,:n_op]).T
    
    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)+np.multiply(1,np.matmul(w_fb,st))
    Ih_t=np.tanh(Ih_temp)+np.multiply(noise_gain_train,np.matrix(np.random.normal(-1,1,(n_rr,1))))

#    Ih_eff=np.row_stack((Ih_t,Io_tm1))
    Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T))
#    Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T,Io_tm1))
    x_train[i,:]=Ih_eff.T   
    y_train[i,:]=st.T
    Ih_tm1=Ih_t
    
print("Time Taken::",time()-curr_time)
#%%
###################################################################
#### Least Error Square: Pinv
###################################################################
y_train=train_output1
w_eff=np.matmul(np.linalg.pinv(x_train),np.arctanh(y_train))       
y_pred_train=np.tanh(np.matmul(x_train,w_eff))

mse_train=np.sqrt(np.mean((np.asarray(y_pred_train-y_train))**2))
print("Train MSE::", mse_train)

fig = plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(train_ctrl[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(y_train[:,0], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(y_pred_train[:,0], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=20)
plt.ylabel('State Variable ::x1', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.title('Golden Output vs. System Output (During Training)', fontsize=24)
plt.legend(fontsize=16)

plt.subplot(2,1,2)
plt.plot(train_ctrl[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(y_train[:,1], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(y_pred_train[:,1], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Variable ::x2', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('Golden Output vs. System Output (During Training)', fontsize=24)
plt.legend(fontsize=10)

#%%
###################################################################
#### Check Pinv on entire signal frame(train + test)
###################################################################
loss_t=0
nmrse_list=[]

test_ctrl1=np.multiply(test_ctrl,ip_scale)+ip_shift
ip_data=np.matrix(test_ctrl1[:,:])
u_t_data=np.matmul(ip_data,w1.T)
test_output1=np.multiply(test_output,op_scale)+op_shift

x_test=np.matrix(np.zeros((len(test_ctrl),n_rr+n_ip)))
y_test=np.matrix(np.zeros((len(test_ctrl),n_op)))
y_test=test_output1
y_test_pred=np.matrix(np.zeros((len(test_ctrl),n_op)))

Io_tm1=np.matrix(y_train[-1,:]).T

#Calculating States
for i in range(len(test_ctrl)):
    
    u_t=u_t_data[i,:].T
    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)+np.multiply(1,np.matmul(w_fb,Io_tm1))
#    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)
    
    Ih_t=np.tanh(Ih_temp)+np.multiply(noise_gain_test,np.matrix(np.random.normal(-1,1,(n_rr,1))))

#   Ih_eff=np.row_stack((Ih_t,Io_tm1))
    Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T))
#   Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T,Io_tm1))
        
#    Io_t=np.matmul(w_eff.T,Ih_eff)
    Io_t=np.tanh(np.matmul(w_eff.T,Ih_eff))
    
    y_test_pred[i,:]=Io_t.T
    
    Io_tm1=Io_t
    Ih_tm1=Ih_t
            
print("Time Taken::",time()-curr_time)        
mse_test=np.sqrt(np.mean((np.asarray(y_test_pred-y_test))**2))
print("Test MSE::", mse_test)

fig = plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(test_ctrl[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(y_test[:,0], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(y_test_pred[:,0], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Variable ::x1', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('Golden Output vs. System Output (During Testing)', fontsize=16)
plt.legend(fontsize=10)               

plt.subplot(2,1,2)
plt.plot(test_ctrl[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(y_test[:,1], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(y_test_pred[:,1], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=12)
plt.ylabel('State Variable ::x2', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('Golden Output vs. System Output (During Testing)', fontsize=16)
plt.legend(fontsize=10)               