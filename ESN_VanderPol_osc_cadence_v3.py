# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 16:09:40 2018

@author: Sarthak
"""

"""
Created on Mon Jul  2 14:57:38 2018

@author: sarthak
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from time import time
from scipy.interpolate import interp1d

np.random.seed(430)

##########################################
##Tuning Curves
##########################################
filename_source_bias="C://Users//Sarthak//Google Drive//IISC Mtech//Machine Learning//Hidden Node Output Curves//Version8//hidden_node_source_sink_bias_source_v8_v2.csv"
data_set=pd.read_csv(filename_source_bias)

v_scale1_start=0.57
v_scale1_stop=0.63
v_scale1_del=0.01
v_scale1_mat=np.matrix(np.arange(v_scale1_start,v_scale1_stop+(v_scale1_del)/2,v_scale1_del))

v_ith_start=0.3
v_ith_stop=0.32
v_ith_del=0.0025
v_ith_mat=np.matrix(np.arange(v_ith_start,v_ith_stop+(v_ith_del)/2,v_ith_del))

v_ibias_start=0.77
v_ibias_stop=0.95
v_ibias_del=0.03
v_ibias_mat=np.matrix(np.arange(v_ibias_start,v_ibias_stop+(v_ibias_del)/2,v_ibias_del))

v_sign1_mat= np.matrix(np.array([0.05,1.18]))

#curr_params=v_ith_mat.shape[1]*v_ibias_mat.shape[1]*v_sign1_mat.shape[1]*2
#curr_params=v_ith_mat.shape[1]*v_ibias_mat.shape[1]*v_sign1_mat.shape[1]*w_cm1_mat.shape[1]*w_cm2_mat.shape[1]*2
curr_params=v_ith_mat.shape[1]*v_ibias_mat.shape[1]*v_sign1_mat.shape[1]*v_scale1_mat.shape[1]*2

##Source Bias
i_in_source=np.flipud(np.multiply(np.matrix(data_set.iloc[1:,0]).T,-1))
i_in_sink=np.matrix(data_set.iloc[:,0]).T
i_in=np.row_stack((i_in_source,i_in_sink))

i_out_mixed=np.matrix(data_set.iloc[:,1:curr_params:2])
v_out_mixed=np.matrix(data_set.iloc[:,curr_params+1::2])
v_out_mixed[v_out_mixed>0.5]=1
v_out_mixed[v_out_mixed<=0.5]=-1

i_out_source=np.flipud(i_out_mixed[1:,:int(curr_params/4)])
i_out_sink=i_out_mixed[:,int(curr_params/4):]

v_out_source=np.flipud(v_out_mixed[1:,:int(curr_params/4)])
v_out_sink=v_out_mixed[:,int(curr_params/4):]

i_out_source1=np.multiply(i_out_source,v_out_source)
i_out_sink1=np.multiply(i_out_sink,v_out_sink)

i_out_bias_src=np.row_stack((i_out_source1,i_out_sink1))

##Sink Bias
filename_sink_bias="C://Users//Sarthak//Google Drive//IISC Mtech//Machine Learning//Hidden Node Output Curves//Version8//hidden_node_source_sink_bias_sink_v8_v2.csv"
data_set=pd.read_csv(filename_sink_bias)

i_in_source=np.flipud(np.multiply(np.matrix(data_set.iloc[1:,0]).T,-1))
i_in_sink=np.matrix(data_set.iloc[:,0]).T
i_in=np.row_stack((i_in_source,i_in_sink))

i_out_mixed=np.matrix(data_set.iloc[:,1:curr_params:2])
v_out_mixed=np.matrix(data_set.iloc[:,curr_params+1::2])
v_out_mixed[v_out_mixed>0.5]=1
v_out_mixed[v_out_mixed<=0.5]=-1

i_out_source=np.flipud(i_out_mixed[1:,:int(curr_params/4)])
i_out_sink=i_out_mixed[:,int(curr_params/4):]

v_out_source=np.flipud(v_out_mixed[1:,:int(curr_params/4)])
v_out_sink=v_out_mixed[:,int(curr_params/4):]

i_out_source1=np.multiply(i_out_source,v_out_source)
i_out_sink1=np.multiply(i_out_sink,v_out_sink)

i_out_bias_sink=np.row_stack((i_out_source1,i_out_sink1))

## Source and Sink Bias
i_out=np.column_stack((i_out_bias_src,i_out_bias_sink))

#Extrapolate
i_in_step=1e-9
num_points_ext= 300
i_in_start_ext= i_in[-1,:]+i_in_step
i_in_ext_pos=np.matrix(np.arange(i_in_start_ext,i_in_start_ext+i_in_step*(num_points_ext-1)+i_in_step/2,i_in_step)).T
i_in_ext_neg=np.flipud(-1*i_in_ext_pos)
i_in=np.row_stack((np.row_stack((i_in_ext_neg,i_in)),i_in_ext_pos))

i_out_pos=np.multiply(np.ones((num_points_ext,i_out.shape[1])),i_out[-1,:])
i_out_neg=np.multiply(np.ones((num_points_ext,i_out.shape[1])),i_out[0,:])
i_out=np.row_stack((np.row_stack((i_out_neg,i_out)),i_out_pos))

np.random.seed(4)
num_nodes=100
sel_nodes=np.array(sorted(np.random.choice(i_out.shape[1],num_nodes,replace=False)))

max_val=np.max(np.abs(i_out[:,sel_nodes]))
act_func={}
i_in1=np.asarray(i_in).reshape(i_in.shape[0],)/max_val
for loop_sel in range(num_nodes):
    io1=np.asarray(i_out[:,sel_nodes[loop_sel]]).reshape(i_out.shape[0],)/max_val
    act_func['func'+str(loop_sel)]=interp1d(i_in1, io1, kind='cubic')

plt.figure(figsize=[8,6])

for loop_sel in range(num_nodes):
#    plt.plot(i_in*1e9,act_func['func'+str(loop_sel)](i_in)*1e9)
    plt.plot(i_in1,act_func['func'+str(loop_sel)](i_in1))
#plt.plot(i_in,i_out[:,0])
#plt.plot(i_in.T,i_in.T,'b')
#plt.legend(['Exact SOUL'],fontsize=18)
plt.xlabel('Input Current (nA)',fontsize=18)
plt.ylabel('Output Current (nA)',fontsize=18)
plt.ylim((-1,1))
plt.xlim((-np.max(np.abs(i_in1)),np.max(np.abs(i_in1))))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.show()    
#%%
def hardware_act_func(I_in):
    l_in=len(I_in)
    num_act_func=len(act_func)
    num_repeats=int(l_in/num_act_func)
    I_out=np.zeros((I_in.shape))
    temp_indent=0
    for loop_act in range(num_act_func):
        I_out[temp_indent:temp_indent+num_repeats,:]=act_func['func'+str(loop_act)](I_in[temp_indent:temp_indent+num_repeats,:])
        temp_indent=temp_indent+num_repeats
    return np.matrix(I_out) 

#%%
###################################################################
####Creating Golden Output and Input
###################################################################
np.random.seed(430)    
tmax=2000
#dt=0.004
dt=0.08
div_unit=1/dt
num_points=div_unit*tmax

t_start=0
#t_end=450

x0,y0=0,1
mu=3

scale=4

def vander_pol(state, t, mu):
    x,y=state
    dxdt=y
    dydt=mu*(1-np.power(x,2))*y-x
    
    return dxdt,dydt

def sim_osc(state, t, mu):
    x,y=state
    dxdt=mu*y
    dydt=-(mu*x)
    
    return dxdt,dydt

###For Vander Pol Oscillator###
t = np.linspace(0, tmax, num_points)

f=odeint(vander_pol,(x0,y0),t, args=(mu,))

f1=f        
golden_out=np.matrix(f1)/np.max(f1[:,:])

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

#%%
###################################################################
#### Train_Test Data
###################################################################

#Using ODEint
traintest_cutoff = int(0.8*len(t))
#traintest_cutoff=20000

#frequency_control=np.column_stack((frequency_control1,frequency_control2))
frequency_control=np.matrix(np.ones((len(t),1)))
#frequency_control=np.column_stack((np.matrix(np.ones((len(t),1))),frequency_control1))

frequency_output1=np.multiply(golden_out[:,0],1)+0.45
frequency_output2=np.multiply(golden_out[:,1],0.5)+0.5
#frequency_output=np.multiply(golden_out[:,0],1)+0.45
frequency_output=np.column_stack((frequency_output1,frequency_output2))

train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]

window_tr = range(int(len(train_output)/4),int(len(train_output)/4+2000))
plt.figure(figsize=(10,4))
plt.plot(t[window_tr],train_ctrl[window_tr,0],label='control1')
#plt.plot(train_ctrl[window_tr,1],label='control2')
plt.plot(t[window_tr],train_output[window_tr],label='target')
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

w2_rr=np.zeros((n_rr,n_rr))
w2_rr_bar=np.zeros((n_rr,n_rr))
 
#Making Sparse Connections in Recurrent Layer
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
w_fb=np.random.uniform(-w_fb_init_range,w_fb_init_range,(n_rr,n_op))

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

#%%
###################################################################
#### Model Implementation :: Offline Learning
###################################################################
st=np.zeros((n_op,1))
curr_time=time()

#ip_scale=np.matrix([2,1])
#ip_shift=np.matrix([0,0])
ip_scale=0.9
ip_shift=0

#op_scale=np.matrix([1.12,1.12])
#op_shift=np.matrix([-0.6,-0.7])
#op_scale=np.matrix([1.12,1.12])
#op_shift=np.matrix([-0.55,-0.65])
op_scale=np.matrix([1.3,1.3])
op_shift=np.matrix([-0.55,-0.65])

train_ctrl1=np.multiply(train_ctrl,ip_scale)+ip_shift

ip_data=np.matrix(train_ctrl1[:,:])
u_t_data=np.matmul(ip_data,w1.T)
train_output1=np.multiply(op_scale,train_output)+op_shift

x_train=np.matrix(np.zeros((len(train_ctrl),n_rr+n_ip)))
y_train=np.matrix(np.zeros((len(train_ctrl),n_op)))

#Calculating States
for i in range(len(train_ctrl)):
    u_t=u_t_data[i,:].T
    if(i>=1):
        st=np.matrix(train_output1[i-1,:n_op]).T
    
    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)+np.multiply(1,np.matmul(w_fb,st))
#    Ih_t=np.tanh(Ih_temp)+np.multiply(noise_gain_train,np.matrix(np.random.normal(-1,1,(n_rr,1))))
    Ih_t=hardware_act_func(Ih_temp)+np.multiply(noise_gain_train,np.matrix(np.random.normal(-1,1,(n_rr,1))))
    
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
num_bits=12
#w_eff=np.matmul(np.linalg.pinv(x_train),np.arctanh(y_train))       
#y_pred_train=np.tanh(np.matmul(x_train,w_eff))
w_eff=np.matmul(np.linalg.pinv(x_train),y_train)       
w_eff_q=np.around(np.multiply(w_eff,np.power(2,num_bits)-1),0)/(np.power(2,num_bits)-1)

y_pred_train=np.matmul(x_train,w_eff_q)

mse_train=np.sqrt(np.mean((np.asarray(y_pred_train-y_train))**2))
print("Train MSE::", mse_train)

fig = plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(t[:traintest_cutoff],train_ctrl1[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(t[:traintest_cutoff],y_train[:,0], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(t[:traintest_cutoff],y_pred_train[:,0], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=16)
plt.ylabel('State Variable ::x1', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('Golden Output vs. System Output (During Training)', fontsize=24)
plt.legend(fontsize=14)

plt.subplot(2,1,2)
plt.plot(t[:traintest_cutoff],train_ctrl1[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(t[:traintest_cutoff],y_train[:,1], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(t[:traintest_cutoff],y_pred_train[:,1], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=16)
plt.ylabel('State Variable ::x2', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('Golden Output vs. System Output (During Training)', fontsize=24)
plt.legend(fontsize=14)

#%%
###################################################################
#### Check Pinv on entire signal frame(train + test)
###################################################################
loss_t=0
nmrse_list=[]

test_ctrl1=np.multiply(test_ctrl,ip_scale)+ip_shift
ip_data=np.matrix(test_ctrl1[:,:])
u_t_data=np.matmul(ip_data,w1.T)
test_output1=np.multiply(op_scale,test_output)+op_shift

x_test=np.matrix(np.zeros((len(test_ctrl),n_rr+n_ip)))
y_test=np.matrix(np.zeros((len(test_ctrl),n_op)))
y_test=test_output1
y_test_pred=np.matrix(np.zeros((len(test_ctrl),n_op)))

Io_tm1=np.matrix(y_train[-1,:]).T

#Calculating System Output
for i in range(len(test_ctrl)):
    
    u_t=u_t_data[i,:].T
    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)+np.multiply(1,np.matmul(w_fb,Io_tm1))
#    Ih_temp=u_t+np.matmul(w2_rr,Ih_tm1)
    
#    Ih_t=np.tanh(Ih_temp)+np.multiply(noise_gain_test,np.matrix(np.random.normal(-1,1,(n_rr,1))))
    Ih_t=hardware_act_func(Ih_temp)+np.multiply(noise_gain_test,np.matrix(np.random.normal(-1,1,(n_rr,1))))

#   Ih_eff=np.row_stack((Ih_t,Io_tm1))
    Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T))
#   Ih_eff=np.row_stack((Ih_t,ip_data[i,:].T,Io_tm1))
        
    Io_t=np.matmul(w_eff_q.T,Ih_eff)
#    Io_t=np.tanh(np.matmul(w_eff.T,Ih_eff))
    
    y_test_pred[i,:]=Io_t.T
    
    Io_tm1=Io_t
    Ih_tm1=Ih_t
          
print("Time Taken::",time()-curr_time)        
mse_test=np.sqrt(np.mean((np.asarray(y_test_pred-y_test))**2))
print("Test MSE::", mse_test)

fig = plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(t[traintest_cutoff:],test_ctrl1[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(t[traintest_cutoff:],y_test[:,0], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(t[traintest_cutoff:],y_test_pred[:,0], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=16)
plt.ylabel('State Variable ::x1', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('Golden Output vs. System Output (During Testing)', fontsize=16)
plt.legend(fontsize=14)               

plt.subplot(2,1,2)
plt.plot(t[traintest_cutoff:],test_ctrl1[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(t[traintest_cutoff:],y_test[:,1], 'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(t[traintest_cutoff:],y_test_pred[:,1], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('Time', fontsize=16)
plt.ylabel('State Variable ::x2', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.title('Golden Output vs. System Output (During Testing)', fontsize=16)
plt.legend(fontsize=14)               

#%%
fig = plt.figure(figsize=(8,4))
#plt.subplot(2,1,1)
#plt.plot(test_ctrl1[:,0], 'g-', label='User Input', linewidth=2, alpha=0.8)
plt.plot(y_test[:,0], y_test[:,1],'b-', label='Desired Output', linewidth=2, alpha=0.8)
plt.plot(y_test_pred[:,0], y_test_pred[:,1], 'r-', label='System Output', linewidth=2, alpha=0.8)
plt.xlabel('State Variable :: y1', fontsize=20)
plt.ylabel('State Variable :: y2', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.title('Golden Output vs. System Output (During Testing)', fontsize=16)
plt.legend(fontsize=18)               
