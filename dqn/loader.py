import os
import glob
import pandas as pd

# Loading the dataset

def load_data(start, end):
    
    months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    main_frame=pd.DataFrame(columns=['Type','Date','Time','O','H','L','C','V'])
    
    for i in range(start,end):
        root=os.path.join('oneminutedata/'+str(i))
        print(root)
        for i in months:
            folder=glob.glob(root+'/'+i+'*')
            print(folder[0])
            
            try:
                dat=pd.read_csv(folder[0]+'/BANKNIFTY.txt',names=['Type','Date','Time','O','H','C','L','V'])
            except FileNotFoundError:
                continue
            
            main_frame=pd.concat([main_frame,dat],ignore_index=True, sort=True)
            data = (main_frame.iloc[:, [0,1,5]]).applymap(str) 
            
            """
            for j in range(len(data)):
                data.iloc[j, 2] = data.iloc[j, 2].replace(":","")     # if using dataset other than that of range [2013, 2018] 
            """               
            data = data.astype(float)
            
    return data
        
