"""
This is the Entry point for Training the Machine Learning Model.

Written By: Soubhagya Nayak

"""

from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data=preprocessor.remove_columns(data,['Wafer']) 

         
            X,Y=preprocessor.separate_label_feature(data,label_column_name='Output')

            
            is_null_present=preprocessor.is_null_present(X)

            
            if(is_null_present):
                X=preprocessor.impute_missing_values(X) 

           
            cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(X)

            
            X=preprocessor.remove_columns(X,cols_to_drop)

            """ Applying the clustering approach"""

            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) 
            number_of_clusters=kmeans.elbow_plot(X)  

            X=kmeans.create_clusters(X,number_of_clusters)

            
            X['Labels']=Y
            list_of_clusters=X['Cluster'].unique()
        
             

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i] 

                
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

               
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder=tuner.Model_Finder(self.file_object,self.log_writer) 

                
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

            
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))

           
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception