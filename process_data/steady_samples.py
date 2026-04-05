#该文件提供了从原始电流/电压信号中提取稳态区间以及计算RMS值的工具函数。

import numpy as np

# fs = 30000：采样率，30 kHz（来自 PLAID 数据集）。
# fn = 60：电网基频，60 Hz。
# dt = 1/fs：采样间隔（秒）。
# n_cycle = fs/fn：每个工频周期的采样点数，即 30000/60 = 500。
fs=30000
fn=60
dt=1/fs
n_cycle=fs/fn

# 通过滑动窗口计算信号的 RMS（均方根）值序列
def generate_rms(signal,mode=None,number_of_cycles=12,sample_frequency=30000,grid_frequency=60):
    n = len(signal)   
    samples_per_cycle=sample_frequency/grid_frequency #每个周期的采样点数
    duration=n/sample_frequency    #信号总时长
    time   = np.linspace(0,duration,n) #长度 n 的等间隔时间点
    signal_rms=np.array([])
    if mode=='half_cycle': 
        resolution=samples_per_cycle/2
    elif mode=='full_cycle':
        resolution=samples_per_cycle
    else:
        resolution=number_of_cycles
    interv=np.arange(0,len(time),resolution) #生成窗口起始索引的数组
    for i in interv:
        signal_pow=0                      
        if (i+resolution)<=(len(time)):#如果当前窗口能完整覆盖（不超界）：
            signal_pow=[signal[j]**2 for j in range(int(i),int(i+(resolution)))]#计算窗口内所有采样点的平方和
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(resolution))]*int(resolution) #计算 RMS 值：sqrt(平方和 / 窗口长度)。
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
        else:
            signal_pow=[signal[j]**2 for j in range(int(i),int(len(time)-i))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(len(time)-i))]*int(len(time)-i) #取剩余的所有点计算 RMS，同样复制该长度后拼接。
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
    return signal_rms

# 寻找最稳态的区间索引，针对 submetered 和 aggregated 数据有不同的处理逻辑。
def get_indices(signal_rms,mode=None,sample_cycles=None,aggregated=0,sample_frequency=30000,grid_frequency=60):
    sample_dict={}
    sample_frag=[]
    n = len(signal_rms) 
    samples_per_cycle=sample_frequency/grid_frequency #计算一个周期的采样点数和窗口步长
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    else:
        resolution=samples_per_cycle
   
    #med_rms=np.mean(signal_rms)
    if sample_cycles==None:
        
        sample_cycles=12
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1): #计算可能的窗口数
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])     
            sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
            #对每个候选窗口，计算窗口内 RMS 的均值 med，然后计算窗口内 RMS 除以均值后的方差（归一化方差），存入字典，键为 (inf, sup)。
        indices=min(sample_dict, key=sample_dict.get) #选择方差最小的窗口作为最稳态区间，
        indices=list(indices)
        return indices
    elif aggregated==0:  #处理 submetered 数据             
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])  #计算均值 med             
            flag=0            
            if med>0.03: #均值大于 0.03，排除噪声底噪
                if all(signal_rms[inf:sup]>0.9*med) and all(signal_rms[inf:sup]<1.1*med):
                    #要求窗口内所有 RMS 值都在均值的 90%~110% 之间
                    sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        if sample_dict!={}:            
            indices=min(sample_dict, key=sample_dict.get) #最后选择方差最小的窗口返回
            indices=list(indices)           
            return indices
        else:
            return None
    else:
        k=0
        while k<=int(n/(n_cycle/2)-sample_cycles):
            inf=int(k*n_cycle/2)
            sup=int(inf+n_cycle*sample_cycles)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0          
            if med>0.01:
                for j in range(2*sample_cycles):   
                    med_local=(signal_rms[inf+(j-1)*(int(n_cycle/2))]+signal_rms[inf+j*(int(n_cycle/2))])/2                         
                    if med_local>1.01*med or med_local<0.99*med:
                        flag=1
                        break
                if flag==0:
                    if sample_frag!=[]:
                        if sample_frag[-1][1]==inf:
                            sample_frag[-1][1]=sup
                        else:
                            sample_frag.append([inf,sup])
                    else:
                        sample_frag.append([inf,sup])
                    k+=2*sample_cycles-1
            k+=1
        print(sample_frag)
        return sample_frag





#用于检测快速电压变化（RVC），依据 IEC 61000-4-30 标准。
def check_rvc(signal_rms,threshold=3.3/100,sample_cycles=120,sample_frequency=30000,grid_frequency=60):
    n_cycle=sample_frequency/grid_frequency
    hysteresis=0.5*threshold    
    event_indexes=[]
    n = len(signal_rms)   
    inf=0
    while inf<=n-sample_cycles*n_cycle/2:
        start_rvc=None
        end_rvc=None    
        start_event=None
        end_event=None
        sup=inf+sample_cycles*n_cycle/2
        if signal_is_steady_state(signal_rms[int(inf):int(sup)])==False: #判断当前窗口是否处于非稳态。若非稳态，则认为可能发生了 RVC。
            start_rvc=int(inf+j*n_cycle/2)
            start_event=start_rvc
            flag=0
            while flag==0 or sup<n:                          
                inf+=n_cycle/2
                sup=inf+sample_cycles*n_cycle/2
                mean_rms=np.mean(signal_rms[int(inf):int(sup)])+0.1                                                             
                if all(signal_rms[int(inf):int(sup)]+0.1<(1+hysteresis)*mean_rms) and all(signal_rms[int(inf):int(sup)]+0.1>(1-hysteresis)*mean_rms) and end_rvc==None:
                    end_rvc=int(inf)                                        
                    if sup>end_rvc+sample_cycles*n_cycle/2:
                        flag=1 
                        end_event=int(sup)                               
            event_indexes.append((start_event,end_event))
        else:
            inf+=n_cycle/2        
    return event_indexes

#判断信号整体是否处于稳态
def signal_is_steady_state(signal,threshold=3.3/100,sample_frequency=30000,signal_frequency=60):    
    signal_rms=generate_rms(signal)
    mean=np.mean(signal_rms)+0.1
    samples_per_cycle=sample_frequency/signal_frequency
     
    n = len(signal_rms)   
    for i in range(int(n/(samples_per_cycle/2))):
        if (signal_rms[i]+0.1>mean*(1+threshold) or signal_rms[i]+0.1<mean*(1-threshold)):
            return False
    return True

    





