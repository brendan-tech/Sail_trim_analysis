"""
Created on Tue Aug 29 22:12:45 2023

@author: brendan
"""

#Import relevant modules
import pylab
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math as m
import pandas as pd




#define functions
def cont_obj_det(mask,min_d,x_len):

    """
    Identify connected objects in a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask to analyze.
        min_d (int): Maximum pixel separation to count as the same object.
        x_len (int): Length of the mask in the x-direction.

    Returns:
        x_ls (list): List of x-coordinates of identified objects.
        y_ls (list): List of y-coordinates of identified objects.
    """
    
    x_ls=[]
    y_ls=[]
    
    x=0
    while x < x_len:
        y_slice=mask[:,x]
        y=0
        for i in y_slice:
            if i == 1:
                #If y_1 is empty add object to the list
                if len(y_ls)==0:
                    x_ls.append(x)
                    y_ls.append(y)
                #If y_1 isnt empty work out if the object should be added to
                #y_1 or y_2
                else:
                    #find minimum seperation to an object in y_1
 
                    x_ls_arr=np.asarray(x_ls)
                    y_ls_arr=np.asarray(y_ls)
                    x_arr=np.full(shape=len(x_ls), fill_value=x)
                    y_arr=np.full(shape=len(y_ls), fill_value=y)
                    d_arr=np.sqrt(np.square(x_ls_arr-x_arr)+
                                  np.square(y_ls_arr-y_arr))
                    d=min(d_arr)
                    if d <= min_d:
                        x_ls.append(x)
                        y_ls.append(y)
            y+=1
                
        x+=1
    
    return(x_ls,y_ls)



def scrub(mask,x_ls,y_ls):
    
    """
    Remove identified objects from a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask to modify.
        x_ls (list): List of x-coordinates of objects to be removed.
        y_ls (list): List of y-coordinates of objects to be removed.

    Returns:
        mask_scrubbed (numpy.ndarray): The modified binary mask.
    """
    
    mask_scrubbed=mask
    c=0
    while c < len(x_ls):
        x=x_ls[c]
        y=y_ls[c]
        mask_scrubbed[y,x]=0
        c+=1
    return(mask_scrubbed)



def average_stripe(x_ls,y_ls,ppl,ew):
    
    """
    Calculate the average coordinates of stripes.

    Args:
        x_ls (list): List of x-coordinates of stripe pixels.
        y_ls (list): List of y-coordinates of stripe pixels.
        ppl (int): Points per line.
        ew (int): Distance from ends of stripe to start and end points.

    Returns:
        x_ls_avg (list): List of averaged x-coordinates.
        y_ls_avg (list): List of averaged y-coordinates.
    """
    
    x_ls_avg=[]
    y_ls_avg=[]
      
    x_min=np.min(x_ls)
    x_max=np.max(x_ls)
        
    #set boundaries
    bounds=[]
    bounds.append(x_min)
    bounds.append(x_min+ew)
    gap=(x_max-ew)-(x_min+ew)
    gap_no=ppl-2
    gap_sec=gap/gap_no
    c=x_min+ew
    while c < (x_max-ew):
        c+=gap_sec
        bounds.append(c)
    bounds.append(x_max)

    z_ls=list(zip(x_ls,y_ls))
    c=0
    while c < ppl:
        xs=[]
        ys=[]
        for i in z_ls:
            if bounds[c]<i[0]<=bounds[c+1]:
                xs.append(i[0])
                ys.append(i[1])
        x_ls_avg.append(np.mean(xs))
        y_ls_avg.append(np.mean(ys))
        c+=1

    return(x_ls_avg,y_ls_avg)



def linear_draft_analysis(x_ls,y_ls,img,col):
    
    """
    Analyze sail shape and calculate camber, draft, and twist.

    Args:
        x_ls (list): List of x-coordinates of stripe pixels.
        y_ls (list): List of y-coordinates of stripe pixels.
        img (numpy.ndarray): The sail image.
        col (str): Color for plotting.

    Returns:
        chord (float): The sail's chord length.
        draft_len (float): The length of the draft.
        m_chord (float): The slope of the chord line.
    """
    
    chord=((x_ls[0]-x_ls[-1])**2+(y_ls[0]-y_ls[-1])**2)**.5
    draft_len=0
    c=0
    while c < len(x_ls)-1:
        d=((x_ls[c+1]-x_ls[c])**2+(y_ls[c+1]-y_ls[c])**2)**.5
        draft_len+=d
        c+=1
    
    sections=1000
    x_start=x_ls[0]
    y_start=y_ls[0]
    x_end=x_ls[-1]
    y_end=y_ls[-1]
    dx=abs(x_end-x_start)/sections
    dy=abs(y_end-y_start)/sections

    m_chord=(y_end-y_start)/(x_end-x_start)
    x=x_start
    y=y_start
    m_perp=-1/m_chord
    
    d_ls=[]
    per_ls=[]
    
    xs=[]
    ys=[]
    
    d=1
    a=0
    while a < sections:
        #Find c-intercept of line perpendicular to chord
        c=y-m_perp*x
        b=0
        while b < len(x_ls)-1:
            m2=(y_ls[b+1]-y_ls[b])/(x_ls[b+1]-x_ls[b])
            c2=y_ls[b]-m2*x_ls[b]
            int_x_tst=(c-c2)/(m2-m_perp)
            int_y_tst=m_perp*int_x_tst+c
            if x_ls[b] <= int_x_tst <= x_ls[b+1]:
                d=(abs(y-int_y_tst)**2+(x-int_x_tst)**2)**.5
                xs.append(int_x_tst)
                ys.append(int_y_tst)
                break
                
            b+=1
        per_ls.append(a/10)
        d_ls.append(d)
        x+=dx
        y+=dy
        a+=1
    
    d_max_index=np.argmax(d_ls)
    d_max=np.max(d_ls)
    d_per_aft=100-(1000-d_max_index)/10
    camber=np.round((d_max/chord)*100,1)
    
    #find start and end points for maximum draft
    x_chrd=x_start+(x_end-x_start)*(d_per_aft/100)
    y_chrd=y_start+(y_end-y_start)*(d_per_aft/100)
    x_draft=xs[d_max_index]
    y_draft=ys[d_max_index]
    
    #plotcamber lines on image 
    plt.imshow(img)
    plt.plot([x_ls[0],x_ls[-1]],[y_ls[0],y_ls[-1]],c=col,
                 linestyle=':',linewidth=3)
    plt.plot([x_chrd,x_draft],[y_chrd,y_draft],c=col,linestyle=':'
             ,linewidth=3)
    plt.plot(x_ls,y_ls,c=col,linewidth=3,label='Camber: '+str(camber)+
             '% Draft: '+str(d_per_aft)+'%',linestyle='-')
    
    print('Camber: '+str(camber)+', Draft % Aft: '+str(d_per_aft))
    return(chord,draft_len,m_chord)


def main():
    
    #read inputs from input file
    df=pd.read_csv('img_inputs.txt',sep=',')
    df['min_d'][0]
    
    #Print Inputs
    print(df)
    
    min_d=df['min_d'][0]            #maximum pixel seperation to count as the
                                    #same object
    p_min=df['p_min'][0]            #minimum percentage of pixels to detect to
                                    #be concidered an object of interest.
    ppl=df['ppl'][0]                #points per line 
                                    #
    ew=df['ew'][0]                    #distance from ends of stripe to start and 
                                    # end points
    stripe_color=df['s_c'][0]       #colour of stripes to analyse 0=orange,
                                    #1=flo_yellow
    filename=df['path'][0]          #path of image to be analysed

    
    #select object for image processing
    img=cv2.imread(filename)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #display image
    pylab.imshow(img)
    pylab.title('Sail image to be analysed')
    pylab.show()

    #define orange range
    ORANGE_MIN = np.array([5, 50, 50],np.uint8)
    ORANGE_MAX = np.array([15, 255, 255],np.uint8)

    #define yellow range
    YELLOW_MIN=np.array([20, 100, 100],np.uint8)
    YELLOW_MAX=np.array([30, 255, 255],np.uint8)

    # Create a mask. Threshold the HSV image to get only selected colors
    if stripe_color == 0:
        mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
    if stripe_color == 1:
        mask = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)    

    mask = np.where(mask!=0, 1, 0)

    #display the mask of the image
    pylab.imshow(mask)
    pylab.title('Pixels identified as colour of intrest')
    pylab.show()

    #note dimensions of picture
    x_len=len(mask[0,:])
    y_len=len(mask[:,0])

    #print image dimensions
    print('image dimensions; x: '+str(x_len)+', y: '+str(y_len))

    #print number of pixels in mask array:
    no=np.sum(mask)
    print('no of interesting pixels: '+str(no)+' out of : '+str(x_len*y_len)+
          ' total pixels. ('+str(np.round(no/(x_len*y_len),3))+'%)')

    #begin timing
    start = time.time()

    #loops to pick out the 3 largest objects in the image with min pixel
    #seperation min_d 
    x_1=[]
    x_2=[]
    x_3=[]
    y_1=[]
    y_2=[]
    y_3=[]

    while len(x_1) == 0:
        x_1,y_1=cont_obj_det(mask,min_d,x_len)
        if len(x_1)*100/no < p_min:
            mask=scrub(mask,x_1,y_1)
            x_1=[]
            y_1=[]
        else:
            mask=scrub(mask,x_1,y_1)

    
    while len(x_2) == 0:
        x_2,y_2=cont_obj_det(mask,min_d,x_len)
        if len(x_2)*100/no < p_min:
            mask=scrub(mask,x_2,y_2)
            x_2=[]
            y_2=[]
        else:
            mask=scrub(mask,x_2,y_2)

        
    while len(x_3) == 0:
        x_3,y_3=cont_obj_det(mask,min_d,x_len)
        if len(x_3)*100/no < p_min:
            mask=scrub(mask,x_3,y_3)
            x_3=[]
            y_3=[]
        else:
            mask=scrub(mask,x_3,y_3)

    print('No of pixels identified: '+str(len(x_1)+len(x_2)+len(x_3)))
    

    #end timing and print
    end = time.time()
    print('Elapsed time for line identification: '+str(end-start)+'s')

    #creating averaged y values for trim stripes
    x_1_avg,y_1_avg=average_stripe(x_1,y_1,ppl,ew)
    x_2_avg,y_2_avg=average_stripe(x_2,y_2,ppl,ew)
    x_3_avg,y_3_avg=average_stripe(x_3,y_3,ppl,ew)

    #Define Title
    plt.title('Identified pixels and mean markers')

    #plot 3 largest objects
    plt.scatter(x_1,y_1,c='r',marker='.',label='line 1 pixels')
    plt.scatter(x_2,y_2,c='b',marker='.',label='line 2 pixels')
    plt.scatter(x_3,y_3,c='g',marker='.',label='line 3 pixels')
    
    #plot averaged points
    plt.scatter(x_1_avg,y_1_avg,c='k',marker='.',label='mean line markers')
    plt.scatter(x_2_avg,y_2_avg,c='k',marker='.')
    plt.scatter(x_3_avg,y_3_avg,c='k',marker='.')
    
    #invert y axis and show
    plt.legend()
    plt.gca().invert_yaxis() 
    plt.show()


    #Calculate Draft and camber and plot
    plt.figure(figsize=(16,12))
    plt.title(filename[:-4]+' shape analysis',fontsize=40)
    c3,d3,mc3=linear_draft_analysis(x_3_avg,y_3_avg,img,'blue')
    c2,d2,mc2=linear_draft_analysis(x_2_avg,y_2_avg,img,'purple')
    c1,d1,mc1=linear_draft_analysis(x_1_avg,y_1_avg,img,'deeppink')

    #Calculate twist and print
    a1=np.tan(mc3)*360/(m.pi*2)
    a3=np.tan(mc1)*360/(m.pi*2)
    twist=np.round(abs(a3-a1),1)
    print('Twist: '+str(twist))

    #plot twist label
    plt.plot(0,0,color='k',linewidth=3,label='Twist: '+str(twist)+' degrees')

    #plot legend, show graph and save graph
    plt.legend(fontsize=20)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(filename[:-4]+'_analysed')
    plt.show()

if __name__ == "__main__":
    main()



