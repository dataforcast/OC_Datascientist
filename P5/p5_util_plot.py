import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)




import p3_util
import p5_util


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_display_linear_regression(df, feature, figsize=(10,10)) :
    f, ax = plt.subplots(figsize=figsize)

    df_droped = df.drop([feature],inplace=False,axis=1)
    df_y = pd.DataFrame(df[feature], index= df_droped.index)

    df_droped = df_droped.astype(float)
    X_std = df_droped.values
    std_scale = preprocessing.StandardScaler().fit(X_std)
    X_std = std_scale.transform(X_std)
    y = df_y.values

    X_train_std, X_test_std, y_train, y_test \
    = model_selection.train_test_split(X_std, y, test_size = 0.8)

    regresion_model = linear_model.LinearRegression()
    regresion_model.fit(X_train_std, y_train.ravel())

    y_predict = regresion_model.predict(X_test_std)

    df_sns_plot=pd.DataFrame(y_test, y_predict).reset_index()
    x_name = 'Measured : '+str(feature)
    y_name = 'Predicted : '+str(feature)

    df_sns_plot.columns=[x_name,y_name]
    df_sns_plot.shape
    df_sns_plot.head()

    sns.regplot(x=x_name, y=y_name, data=df_sns_plot, ax=ax)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_df_subplot(df,type_plot='dist', is_outliers_removed = True\
               , tuple_reshape=(2,3), bins_parameter=50) :

    f, axes = plt.subplots(tuple_reshape[0], tuple_reshape[1], figsize=(14, 14), sharex=True)
    arr_column = np.array(df.columns)
    
    #---------------------------------------------------------------------------
    # List of columns is upgraded for matching with 
    # tuple_reshape[0] * tuple_reshape[1]
    #---------------------------------------------------------------------------
    len_arr_column = len(arr_column)
    delta_len = tuple_reshape[0]*tuple_reshape[1] - len_arr_column
    
    for i in range(0,delta_len):
      arr_column = np.append(arr_column,'None')

    if 0 > delta_len:
      len_arr_column += delta_len
      arr_column = arr_column[0:len_arr_column]  
    
    arr_column = np.reshape(arr_column,tuple_reshape)
    list_basic_color= ['skyblue','olive','gold','red','pink','black']
    

    n_color = tuple_reshape[0]*tuple_reshape[1]
    # Increment elements from list_color until reaching number of diagrams
    for i_color in range(0,n_color):
       if n_color > len(list_basic_color):
         list_basic_color +=list_basic_color  
       else:
         break   
        
    #---------------------------------------------------------------------------
    # List color is built
    #---------------------------------------------------------------------------        
    list_color = list()
    for i_color in range(0,n_color,1) :
      color = list_basic_color[i_color]
      list_color.append(color)

    list_color = np.reshape(list_color,tuple_reshape)
    axes = np.reshape(axes,tuple_reshape)

    for row in range(0,arr_column.shape[0]) :
        for col in range(0,arr_column.shape[1]) :
            feature = arr_column[row,col]
            if feature == 'None':
               pass
            else:
               # Remove outliers from plot
               if is_outliers_removed is True :
                   q1,q3,zmin,zmax = p3_util.df_boxplot_limits(df , feature)
                   df_plot = df[df[feature]<q3]
                   df_plot = df_plot[df_plot[feature]>q1]
               else :
                   df_plot = df.copy()
               if type_plot == 'dist' :
                   sns.distplot( df_plot[feature] , color=list_color[row, col]\
                   , ax=axes[row, col], bins=bins_parameter)
               else :
                   sns.boxplot( y=df_plot[feature], ax=axes[row, col])

    plt.show()
    plt.clf()
    plt.close()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_2D_dict_tsne_result(dict_dbscan_result, nb_col, ratio=1.0\
    , annotation=None):
    """Plot 2D results issue from dict_dbscan_result dictionary.
    Plotting is expanded over nb_col and multiple rows. 
    Each row contain nb_col diagrams.
    """
    
    len_dict = len(dict_dbscan_result)
    max_row = (int(len_dict/nb_col)+(min(1,len_dict%nb_col)))
    row = 0
    col = 0
    f, ax = plt.subplots(max_row, nb_col, sharex='col', sharey='row'\
    ,figsize=(20,20))
        
        
    
    for perplexity, X in dict_dbscan_result.items():
        # col variable is ranged from 0 to nb_col        
        if col == nb_col:
            row += 1
            col = 0
        else :
            pass

        #-----------------------------------------------------------------------
        # Randomly get nb_index
        #-----------------------------------------------------------------------
        size=len(X)
        nb_index = int(size*ratio)
        index_array = np.random.randint(0,size,nb_index)
        
        #-----------------------------------------------------------------------
        # Select part of array sliding original array with random indexes
        #-----------------------------------------------------------------------
        X = X[index_array]

        if isinstance(ax, np.ndarray) is False:
            ax.scatter(X[:, 0], X[:, 1], c='grey', alpha=0.6)
            ax.set_title("Perplexity= "+str(perplexity),color='blue')
            ax.grid(True)
            if annotation is not None:
                annotation = annotation[index_array]
                for i in range(0,len(index_array)):
                    ax.annotate(annotation[i], (X[i, 0], X[i, 1]))
        else:
            ax[row,col].scatter(X[:, 0], X[:, 1], c='grey', alpha=0.6)
            ax[row,col].set_title("Perplexity= "+str(perplexity),color='blue')
            ax[row,col].grid(True)
        col += 1
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_kmeans_interInertia(dict_kmeans, cluster_start, cluster_end\
                             ,list_x_label, p_rows=1, p_cols=2\
                             , p_figsize=(10,10), is_inertia=True):
   """Plot Kmeans inter-intertia values for each cluster in the given 
   dict_kmeans
   Display 2 graphics :
      a) Inter-inertia values = F(nb clusters)
      b) Inter-inertia differences values = F(nb clusters)
      These last graphic allows to visualize the nb. clusters limit.
   Input : 
      * dict_kmeans : dictionary of (keys, values)=(i,kmeans) values
      * p_rows : number of graphic rows for dispatching graphics
      * p_cols : number of graphics columns for dispatching graphics
   Output : none            
   """
   #----------------------------------------------------------------------------
   # Get inter-inertia  values for each cluster from clusters dictionary
   #----------------------------------------------------------------------------
   list_inter_inertia = list()
   for i in dict_kmeans.keys():
       list_inter_inertia.append(dict_kmeans[i].inertia_)

   #----------------------------------------------------------------------------
   # Compute inter-intertia difference between 2 consecutives clusters
   #----------------------------------------------------------------------------
   list_gap_cluster = [ np.abs(val-list_inter_inertia[i+1]) \
   for i,val in zip(range(0,len(list_inter_inertia)-1), list_inter_inertia) ]

   list_gap_cluster.append(0)


   #----------------------------------------------------------------------------
   # Get graphics areas
   #----------------------------------------------------------------------------
   f, ax = plt.subplots(p_rows, p_cols, figsize=p_figsize, sharex=True)
   list_display = [list_inter_inertia, list_gap_cluster]
   
   #----------------------------------------------------------------------------
   # Display graphics inside each areas
   #----------------------------------------------------------------------------
   if p_cols >=2 :
       list_title = ['Inerties internes', 'Variation inerties internes']
       for col in range(0,p_cols):
          ax[col].plot(range(cluster_start,cluster_end+1),list_display[col])
          ax[col].scatter(range(cluster_start,cluster_end+1),list_display[col])
          ax[col].set_title(list_title[col],color='blue')
          ax[col].set_xlabel(list_x_label[col])
          ax[col].set_xticklabels(range(cluster_start*10,(cluster_end+1)*10))
          ax[col].grid(True)
   else :
       col=0
       list_title = ['Inerties internes']
       list_display = [list_inter_inertia]
       ax.plot(range(cluster_start,cluster_end+1),list_display[col])
       ax.scatter(range(cluster_start,cluster_end+1),list_display[col])
       ax.set_title(list_title[col],color='blue')
       ax.set_xlabel(list_x_label[col])
       ax.set_xticklabels(range(cluster_start*10,(cluster_end+1)*10))
       ax.grid(True)
    

   plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_kmeans_silhouette(dict_cluster_scoring, p_title, p_figsize=(7,7)):
   """Display graphics of function Silhouette = F(nb clusters).
   Input :
      * dict_cluster_scoring : (keys,values)=(clusterID,Silhouette)
   Output : none
   """
   cluster_start = list(dict_cluster_scoring.keys())[0]
   cluster_end = list(dict_cluster_scoring.keys())[len(dict_cluster_scoring)-1]

   f, ax = plt.subplots(1, 1, figsize=p_figsize, sharex=True)

   ax.plot(dict_cluster_scoring.keys(),dict_cluster_scoring.values())
   ax.scatter(dict_cluster_scoring.keys(),dict_cluster_scoring.values())
   ax.set_title(p_title,color='blue')
   ax.set_xlabel('Nb. clusters')
   ax.set_xticklabels(range(0, len(dict_cluster_scoring)+1))
   ax.grid(True)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def plot_cluster_frequency(df, p_title=None):
   """Displays clusters frequency from inside a dataframe as bar.
   Column with cluster name has to be present into dataframe.
   
   Input : 
      * df : dataframe containing clusters to display as bars.
      
   Output :  none
   """
   if 'cluster' not in df.columns:
      print("*** ERROR : no name \'cluster\' as column!")
      return
   else:
      pass

   print("Clusters = "+str(np.unique(df.cluster.values)))

   df_ = df.groupby('cluster').agg({'cluster':lambda x: len(x)})
   df_.rename(columns={'cluster':'Count'},inplace=True)

   print("Population cumulée par cluster = "+str(df_.Count.sum()))

   if p_title is not None:
      p_title += ' : Effectif par cluster'
   else:
      p_title = 'Effectif par cluster'
   ax = df_.plot.bar(figsize=(7, 7), title=p_title, color='blue', fontsize=14)
   #ax = df_.plot.bar(figsize=(7, 7), title=p_title, color='blue', fontsize=14, kind='kde')
   print(df_.sort_values(by=['Count'], ascending=False))
   return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_models_plot_AIC_BIC(df, dict_list_gmm_model\
                           , t_range_cluster, p_figsize, p_title):
   """For each hyper parameter values, plot, into 2 separated areas, 
   AIC and BIC curves depending on GMM models. 
   
   Input : 
      * df : dataframe containing values from which, GMM models are built then 
      evaluated against AIC and BIC criterias.
      * dict_list_gmm_model : dictionary of GMM models per hyper-parameter.
      Each hyper-parameter addresses a list of models, ranking from 
      cluster_start to cluster_end.
      * t_range_cluster : cluster ranks
      * p_figsize : tuple for figure sizing.
      * p_title : graphic title
   Output : none
   """


   X = df.values
   n_components = range(t_range_cluster[0], t_range_cluster[1])
   
   plt.figure(figsize=p_figsize)
   
   f, ax = plt.subplots(1, 2, figsize=p_figsize, sharex=True)

   for hyper_param_type, list_gmm_model in dict_list_gmm_model.items():

      #----------------------------------------------------------------------------
      # Display graphics inside each areas :
      # 1st area : AIC
      # 2nd area : BIC
      #----------------------------------------------------------------------------
      for col in range(0,2):
         if 0 == col :
            ax[col].plot(n_components\
            , [model.aic(X) for model in list_gmm_model]\
            , label='AIC: '+str(hyper_param_type))
         else :
            ax[col].plot(n_components\
            , [model.bic(X) for model in list_gmm_model]\
            , label='BIC: '+str(hyper_param_type))
         
         ax[col].set_title(p_title,color='blue')
         ax[col].set_xlabel('Nb clusters', color='blue')
         ax[col].legend(loc="best")

   return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_models_plot_AIC_BIC_deprecated(df, dict_list_gmm_model\
                           , t_range_cluster, p_figsize, p_title):
   """For each hyper parameter values, plot AIC and BIC curves depending on
   GMM models. 
   Plot are performed on a single area.
   
   Input : 
      * df : dataframe containing values from which, GMM models are built then 
      evaluated against AIC and BIC criterias.
      * dict_list_gmm_model : dictionary of GMM models per hyper-parameter.
      Each hyper-parameter addresses a list of models, ranking from 
      cluster_start to cluster_end.
      * t_range_cluster : cluster ranks
      * p_figsize : tuple for figure sizing.
      * p_title : graphic title
   Output : none
   """
   X = df.values
   n_components = range(t_range_cluster[0], t_range_cluster[1])
   plt.figure(figsize=p_figsize)

   for hyper_param_type, list_gmm_model in dict_list_gmm_model.items():

      #-------------------------------------------------------------------------
      # For any GMM model from list_gmm_model, compute and plot both BIC and AIC 
      #-------------------------------------------------------------------------
      plt.plot(n_components, [model.bic(X) for model in list_gmm_model]\
      , label='BIC: '+str(hyper_param_type))

      plt.plot(n_components, [model.aic(X) for model in list_gmm_model]\
      , label='AIC: '+str(hyper_param_type))

      plt.legend(loc='best')
      plt.title(p_title, color='blue')
      plt.xlabel('Nb clusters', color='blue');

   return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_models_plot_silhouette(df, dict_dict_silhouette_score\
, p_figsize, p_title, areas_raws, areas_colums):
   """Plot silhouette scores for any GMM models and for any hyper-parameter.
   GMM models are ranking from cluster_start to cluster_end.
   Hyper-parameters are keys from dict_dict_silhouette_score.
   Input : 
      * df : dataframe containing values from which, GMM models are built then 
      evaluated against silhouette coefficient.
      * dict_dict_silhouette_score : dictionaty of dictionaries.
      Keys dictionary of dict_dict_silhouette_score are GMM hyper-parameters.
      Values dictionaries of dict_dict_silhouette_score are dictionaries 
      from which  :
         --> keys are clusters
         --> values are silhouette scores for a cluster.
      * p_figsize : tuple, for sizing pyplot figure.
      * p_title : figure title
      * areas_raws : number of raws against witch figures areas are expanded
      * areas_colums : number of columns against witch figure is expanded
   Output : none
   """

   f, ax = plt.subplots(areas_raws, areas_colums, figsize=p_figsize\
   , sharex=True, sharey=True)


   range_area = range(areas_raws-1,areas_colums)
   for hyper_parameter,col_area in zip(dict_dict_silhouette_score,range_area):
       dict_silhouette_score = dict_dict_silhouette_score[hyper_parameter]    
       if 1 == areas_colums:
        ax_area = ax
       else:
        ax_area = ax[col_area]
       #------------------------------------------------------------------------
       # For any GMM model from list_gmm_model, plot both BIC and AIC 
       #------------------------------------------------------------------------
       ax_area.plot(dict_silhouette_score.keys()\
       , dict_silhouette_score.values(), label=str(hyper_parameter))
       ax_area.scatter(dict_silhouette_score.keys()\
       , dict_silhouette_score.values(), label=str(hyper_parameter))
       ax_area.legend(loc='best')
       
       ax_area.set_title(p_title, color='blue')
       ax_area.set_xlabel('Nb clusters', color='blue');

   return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_print_median_cluster(df, col, list_cluster):
   """ Prints median values per cluster for column col belonging to 
   dataframe df.
   
   clusters values from where median values are printed belongs to list_cluster.
   
   Input :
      * df : dataframe from which median values are computed
      * col: column for median values
      * list_cluster : list of clusters values median values belongs to.

   Output : none
   """
   if col not in df.columns:
      print("*** ERROR : no column= "+str(col)+" in given dataframe!")
      return
   if 'cluster' not in df.columns:
      print("*** ERROR : no column= cluster in given dataframe!")
      return

   dict_col_median = dict()
   for cluster in list_cluster:
       median = df[df.cluster==cluster][col].median()
       index= 'Cluster '+str(cluster)
       dict_col_median[index] = median
   
   df_col_median = pd.DataFrame(dict_col_median, index=['median',])
   print(df_col_median)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_plot_feature_cluster(df, df_customers, feature,new_feature, list_cluster\
, lambda_func, plot_cond, y_label, title, hue=None):

   """Display a feature as violinplot or boxplot for each cluster.
   The feature given as parameter in aggrgated per customer and the 
   aggragate function lambda_func is applied to the aggregated feature.
   
   """
   if 'CustomerID' not in df.columns :
      print("*** ERROR : no column= CustomerID into dataframe!")
      return

   if feature not in df.columns :
      print("*** ERROR : no column= "+str(feature)+" into dataframe!")
      return

   # ------------------------------------------------------------------------
   # Creates df_cluster with per customer
   # ------------------------------------------------------------------------
   df_feature_cluster = df.groupby(['CustomerID']).agg({feature: lambda_func})

   df_feature_cluster.rename(columns={feature:new_feature}, inplace=True)

   # ----------------------------------------------------------
   # Reset index leads to CustomerID as a column
   # ----------------------------------------------------------
   df_feature_cluster.reset_index(inplace=True)

   # ------------------------------------------------------------------------
   # Clusters from list_cluster are assigned to each customer
   # ------------------------------------------------------------------------
   df_feature_cluster = p5_util.df_cluster_list_add(df_feature_cluster\
   , df_customers, list_cluster)

   # ------------------------------------------------------------------------
   # Plot applying condition over data from df_feature_cluster
   # ------------------------------------------------------------------------      
   if len(plot_cond) >0 :
      sns.violinplot(y=new_feature, x='cluster', 
                       data=df_feature_cluster.query(plot_cond), 
                       width=0.8,
                       inner='quartile',
                       palette="colorblind")
   else:
      sns.violinplot(y=new_feature, x='cluster', 
                       data=df_feature_cluster, 
                       width=0.8,
                       inner='quartile',
                       palette="colorblind")

   plt.xlabel('Clusters')
   if y_label is not None:
      plt.ylabel(y_label)
   plt.title(title, fontsize=14, color='blue')
   plt.grid(True)
   
   plt.plot()


   # ------------------------------------------------------------------------
   # Median value is printed
   # ------------------------------------------------------------------------
   df_print_median_cluster(df_feature_cluster,new_feature\
   , list_cluster)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_monthly_incomes_plot(df, dict_month_name, list_cluster):
   """ Build dataframe with rows as clusters and  months as columns..
   
   Input :
      * df : dataframe containing monthly incomes formated as following : 
        --> First column : cluster
        --> Other columns : Jan, ..., Dec
      * dict_month_name : dictionary relating month value to month name.
      * list_cluster : lits of clusters to be plot.

   Output : none
   """

   #----------------------------------------------------------------------------
   # Build dataframe where incomes are computed per cluster : initialization
   #----------------------------------------------------------------------------
   df_cluster_incomes = df.groupby('cluster').agg({'Jan': lambda x:sum(x)})
   min_month = 1
   max_month = 12
   for month in range(min_month+1, max_month+1):
      #-------------------------------------------------------------------------
      # Compute incomes per month for each cluster (grouped per cluster)
      #-------------------------------------------------------------------------
      month_name = dict_month_name[month]
      df_ = df.groupby('cluster').agg({month_name: lambda x:sum(x)})
      df_cluster_incomes = pd.concat([df_cluster_incomes,df_], axis=1\
      , join='inner')
    

   df_cluster_incomes_pivoted = pd.DataFrame()
   for cluster in list_cluster:
      df_cluster_incomes_pivoted.loc[:,cluster] = df_cluster_incomes.loc[cluster,:]

   #-------------------------------------------------------------------------
   # Cluster colums are renamed in a human readable way
   #-------------------------------------------------------------------------
   dict_rename_cluster = dict()
   for cluster in list_cluster:
       dict_rename_cluster[cluster] = 'Cluster '+str(cluster)

   df_cluster_incomes_pivoted.rename(columns=dict_rename_cluster, inplace=True)

   #-------------------------------------------------------------------------
   # This function allows to format ticks on X axe.
   #-------------------------------------------------------------------------
   def format_func(value, tick_number):
       return dict_month_name[value+1]

   #-------------------------------------------------------------------------
   # Plot
   #-------------------------------------------------------------------------
   axes = df_cluster_incomes_pivoted.plot(figsize=(10,10))

   axes.set_title('Total incomes per month for each cluster', fontsize=14\
   , color='blue')
   axes.set_ylabel('Incomes')

   axes.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11])
   axes.xaxis.grid()
   axes.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

#-------------------------------------------------------------------------------

