
#author- najeeb qazi

import re
import pandas as pd
import math

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)
         
    def removeEmptystr(self,row):
        emptyList= []
        for val in row:
            if val != "":
                emptyList.append(val)
        return emptyList
        
    def preprocess_df(self, df, cols): 
        
        df = df.fillna('')
        df['joinkey'] = df[cols].apply(lambda row : " ".join(row.values.astype(str)), axis = 1 )
        df['joinkey'] = df['joinkey'].apply(lambda row : re.split(r'\W+', str(row).lower()))
        #remove empty strings from Dataframe
        df['joinkey'] = df['joinkey'].apply(self.removeEmptystr)
        return df
    
    #splits listed columns into single rows
    def splitListToRows(self,row,row_accumulator,target_column,new_column):
        #create a new record with all the other values duplicated and one value from the joinkey column
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[new_column] = s
            row_accumulator.append(new_row)
    
    def filtering(self, df1, df2):
        
        new_rows = []
        df1.apply(self.splitListToRows,axis=1,args = (new_rows,"joinkey","flatten"))
        new_df1 = pd.DataFrame(new_rows)
        new_rows = []
        df2.apply(self.splitListToRows,axis=1,args = (new_rows,"joinkey","flatten"))
        new_df2 = pd.DataFrame(new_rows)
        
        df_inner = pd.merge(new_df1, new_df2, on='flatten', how='left')
        df_inner = df_inner.drop(['flatten'],axis =1)
        df_inner.drop_duplicates(subset=["id_x","id_y"], keep = "first", inplace = True) 
        df_inner = df_inner[['id_x','joinkey_x','id_y','joinkey_y']]
        df_inner = df_inner.rename(columns={"id_x": "id1", "joinkey_x": "joinKey1", "id_y":"id2","joinkey_y":"joinKey2"})
        df_inner = df_inner.dropna()
        return df_inner

    #returns the jaccard similarity
    def computesimilarity(self,row):
        new_dict={}
        for item in row:
            if(item):
                if item not in new_dict:
                    new_dict[item] = 1
                else:
                    new_dict[item] += 1

        rUs = len(new_dict.keys())
        
        return rUs
                
    #returns the intersection
    def commonElements(self,row):
        return len(set(row['joinKey1']) & set(row['joinKey2']))
    
    def verification(self, cand_df, threshold):
        
        #concate the joinkey columns
        cand_df['concatedCol'] = cand_df['joinKey1'] + cand_df['joinKey2']
        #calculate the unique elements in the concatenated col to get r U s
        cand_df['rUs'] = cand_df['concatedCol'].apply(self.computesimilarity)
        #find the intersection of r and s
        cand_df['common'] = cand_df.apply(self.commonElements,axis=1)
        cand_df['jaccard'] = cand_df['common']/cand_df['rUs']
        
        cand_df = cand_df[cand_df['jaccard'] >= threshold ]
        cand_df = cand_df.drop(['rUs','common','concatedCol'],axis =1)
        return cand_df
        
    def evaluate(self, result, ground_truth):
        
        R = len(result)
        T= 0
        for item in result:
            if item in ground_truth:
                T += 1
        precision = T/R
        
        A = len(ground_truth)
        recall = T/A
        fmeasure = (2*precision*recall)/(precision + recall)
        return (precision, recall, fmeasure)
                
        
    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))
        
        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))
        
        return result_df
    
 
        

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
