from math import ceil
import numpy as np
def norm_mat(a):
    """
    normalize the array a
    """
    a= np.array(a)
    b =(a - np.min(a))/( np.max(a) - np.min(a))
    return b
def generateCGI_func(img_ori,pattern , beta , sizex=54 , sizey=98):
    """
    generate ghost images from image_ori
    """
    
    # % 预先加载pattern and global txt
    # % CGIpic is normalized, and target is range from 0 to 255
    # % 1 白噪声正则
    # % 2 白噪声
    # % 3 红噪正则
    # % 4 红噪声
    # % the CGIpic is normalized to [0,1]



    img_ori = np.array(img_ori)
    Npixel=sizex*sizey
    Nth=ceil(beta*Npixel) 


    PI=np.zeros([sizex,sizey])
    P=np.zeros([sizex,sizey])
    I=np.zeros([Nth]);
    for j in range(Nth):
        txt = pattern.swapaxes(2,0)[:][:][j].swapaxes(1,0)
        pa=img_ori*txt
        I[j]=np.sum(pa)
        # %     P=P+eval(['txt',num2str(j),'(1:sizex,1:sizey)']);
        P=P+txt
        # %     PI=PI+eval(['I(',num2str(j),').*txt',num2str(j),'(1:sizex,1:sizey)'])
        PI=PI+I[j]*txt

    PI_mean=PI/Nth;
    P_mean=P/Nth;
    I_mean=np.sum(I)/Nth;
    result=PI_mean-(P_mean*I_mean);
    # % eval(['image_input',num2str(i),'=result;']);
    CGIpic=norm_mat(result);
    # % CGIpic=result;



    return CGIpic