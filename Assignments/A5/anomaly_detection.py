#author- najeeb qazi

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ast import literal_eval


class AnomalyDetection():
    
        
    def scaleNum(self, df, indices):
         
        #converting values to np array    
        new_df= np.asarray(df.features.values.tolist()).astype(np.float)
        #using vectorized operations to find mean and std
        for pos in indices:
            mean_val = new_df[:,pos].mean()
            std_val = new_df[:,pos].std(ddof=1)
            new_df[:,pos] = (new_df[:,pos] - mean_val)/(std_val)
            
            
        dat = pd.DataFrame({'features_scaled': new_df.tolist()})
        df['features']= dat['features_scaled']
        return df


    def cat2Num(self, df, indices):
        
        #converting lists to df to find unique
        new_df= pd.DataFrame(df.features.values.tolist())
        new_list = []
        #finding unique features in the indices given
        for val in indices:
            new_list.extend(new_df[val].unique())
            
        target_col= np.asarray(df.features.values.tolist())
        #making a vector of zeros based on length of df and unique features
        row_vec = np.zeros((len(df),len(new_list)))
        rest_of_features = target_col[:,len(indices):]
        #making one hot encoding based on the features
        for row in range(len(df)):
            for val in indices:
                if(target_col[row][val] in new_list):
                    row_vec[row][new_list.index(target_col[row][val])] =1
        #concatenating with the rest of the features
        row_vec = np.concatenate((row_vec,rest_of_features),axis=1)
        dat = pd.DataFrame({'features_oneHot': row_vec.tolist()})
        df['features'] = dat['features_oneHot']
        return df



    def detect(self, df, k, t):
        
        X = np.array(df['features'].values.tolist())  
        #kmeans = KMeans(n_clusters=k,init='k-means++').fit(X)
        kmeans = KMeans(n_clusters=k,random_state=42).fit(X)
        df['prediction'] =kmeans.predict(X)
        
        #grouping by predictions to get cluster counts
        df_grouped = df.groupby(['prediction'],as_index=False).count()
        
        #join grouped by and original dataframe
        df = pd.merge(df,df_grouped,how="inner",on="prediction",suffixes=('_x', '_y'))
        preds = df_grouped['prediction'].values
        value_counts = df_grouped['features'].values
        cluster_counts = {}
        for i in range(len(preds)):
            cluster_counts[preds[i]] = value_counts[i]
        #find min and max cluster size
        max_cluster_size = max(cluster_counts.values())
        min_cluster_size = min(cluster_counts.values())
        #get score values
        df['score'] = (max_cluster_size - df['features_y'])/(max_cluster_size-min_cluster_size)
        df = df[df['score'] >= t]
        df = df.rename(columns={'features_x':'features'})
        df = df.drop(['prediction','features_y'],axis=1)
        return df



if __name__ == "__main__":
#     data = [(0, ["http", "udt", 4]),
#             (1, ["http", "udf", 5]),
#             (2, ["http", "tcp", 5]),
#             (3, ["ftp", "icmp", 1]),
#             (4, ["http", "tcp", 4])]
    df = pd.read_csv('A5-data/A5-data/logs-features-sample.csv').set_index('id')
    df['features'] = df['features'].apply(literal_eval)
#     df = pd.DataFrame(data=data, columns = ["id", "features"])
    
    ad = AnomalyDetection()
    
    df1 = ad.cat2Num(df, [0,1])
    print(df1)
    #scaled the 13th feature for logs_sample_csv
    df2 = ad.scaleNum(df1, [13])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)
