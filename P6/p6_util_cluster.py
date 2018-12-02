import pandas as pd
import numpy as np
import operator

import p6_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_kmeans_get_cluster_from_post(cluster_kmean, vectorizer, df, doc_id):
    """Returns cluster ID POST belongs to and vectorized POST from a post 
    issue from test dataframe.
    Input 
        * cluster_kmean : model issued from kmeans clustering
        * df_sof_test : dataframe POST given as parameter belongs to.
        * doc_id : document identifier
    Output : 
        * cluster_id : cluster identifier issue from Kmeans model 
        doc_id belongs to.
        * X : vectorized document identifier by doc_id
    """
    body  = df.Body.iloc[doc_id]
    title = df.Title.iloc[doc_id]
    post  = body+title

    csrmatrix = p6_util.p6_get_vectorized_doc(vectorizer, post)
    cluster_id = cluster_kmean.predict(csrmatrix)[0]
    return cluster_id, csrmatrix.toarray()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_kmeans_dist_point(cluster_kmean, cluster_id, X) :
    """Returns the distance from a vector belonging to a Kmeans cluster and the 
    cluster centroid.
    Input : 
        * cluster_kmean : Kmeans clustering model
        * cluster_id : cluster identifier from which centroid distance is 
        computed
        * X : vector beloging to the cluster identifie by cluster_id
    """
    # Calcul de la distance an centroid
    X_centroid = cluster_kmean.cluster_centers_[cluster_id]
    dist = np.linalg.norm(X - X_centroid.T.reshape(1,-1))

    return dist
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_cluster_get_list_dist_with_matrix(cluster_model, cluster_id\
    , X_corpus, cluster_type='kmeans'):
    
    '''Returns a dictionary of all distances between a subset of a matrix 
    of vectors and the center of a cluster. 
    Sub set of matrix vectors represents a set of documents belonging to 
    the cluster.
    
    Cluster center definition depends on the cluster type.
    
    Input :
        * cluster_model : model of cluster distances will be computed from.
        * cluster_id : cluster identifier from which all distances 
        will be computed.
        * X_corpus : a vectorized corpus.
        * cluster_type : type of clustering algorithm used for building 
        cluster_model.
    Output :
        * list of tuples, each tuple structured as following : 
        (document_id, distance) where :
            --> document_id belongs to the vectorized corpus 
            --> distance is the distance between the document and the 
            cluster center.
    '''
    
    dict_cluster_dist = dict()
    if 'kmeans' == cluster_type :
        #-----------------------------------------------------------------------
        # Array of indexes of documents belonging to cluster_id
        #-----------------------------------------------------------------------
        arr_index_doc_cluster = np.where(cluster_model.labels_==cluster_id)[0]
        
        #-----------------------------------------------------------------------
        # Distance is computed for each document identifier from 
        # arr_index_doc_cluster.
        # Distances are stored in a dictionary
        #-----------------------------------------------------------------------
        for index_doc_cluster in arr_index_doc_cluster :
            X = X_corpus[index_doc_cluster]
            dict_cluster_dist[index_doc_cluster] \
            = p6_kmeans_dist_point(cluster_model, cluster_id, X.toarray())

    else :
        print("\n*** ERROR : cluster model= "+str(cluster_type)\
        +" not yet supported!")
        return None

    #---------------------------------------------------------------------------
    # Dictionary values are sorted by reversed valeus; result is stored in a 
    # list of tuples each tuple is structured as following : (document_id, value)
    # Sort apply on value.
    #---------------------------------------------------------------------------
    list_sorted_dist = sorted(dict_cluster_dist.items()\
    , key=operator.itemgetter(1), reverse=True)

    return list_sorted_dist
#-------------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_cluster_get_position_distance_from_list(list_tuple_dist, dist) :
    '''Returns a position in a list of distances for a given distance.
    Given distance does not belongs to the list of distances.
    
    List contains ordered distances, reversed order.
    
    In the scheme below, where cluster_center is the point from which all 
    distances are computed, then pos#k is returned.
    cluster_center < dist#0 ...< dist#k < dist  <  dist#k+1 < ... < dist#N
                        ^         ^        ^          ^                ^      
                        |         |        |          |                |
                      pos#0     pos#k     pos#dist   pos#k+1        pos#N
     
    Input :
        * list_tuple_dist : list of tuples containing distances to the cluster 
        center, each tuple in list is structured as following: 
        (doc_id, distance)
        * dist: float, distance from cluster center computed for a point into 
        the cluster.
    '''
    position_in_list=0
    for tuple_dist in list_tuple_dist :
        if tuple_dist[1] >= dist :
            position_in_list += 1
            continue
        else :
            break
    return position_in_list
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_cluster_get_suggested_tag(cluster_model, vectorizer, csr_matrix_corpus\
                              , df_corpus, doc_id, cluster_type='kmeans'\
                              , neighborhood=3, cross_cluster=True\
                              , verbose=False) :
    '''Returns, from a cluster, a list of suggested TAG for a document.
    A cluster is assigned to the document.

    
    Inputs :
        * cluster_model : model of clustering from which clusters are issued.
        * vectorizer : operator used for representing a corpus as a matrix.
        * csr_matrix_corpus : vectorized corpus
        * df_corpus : dataframe document belongs to.
        * doc_id : document identifier into df_corpus.
        * cluster_type : cluster algorithm used for building cluster_model
        * neighborhood : list of documents from which TAG will be picked 
        as assigned TAG will be in the range of [pos-window, pos+window] 
        where pos is the position of doc_id in the list of distances, 
        reverse order. List of distances are distances of any document in 
        cluster doc_id belongs to and cluster center.
        * cross_cluster : when True, suggested TAG are searched among other 
        documents in cluster. Otherwise, suggested TAG are searched in doc_id 
        only.
        * verbose : when True, some variables are displayed along with process.
        
        
    '''
    if ('kmeans'==cluster_type):
        #-------------------------------------------------------------------------
        # Get cluster ID a POST belongs to and a vector representing the POST
        #-------------------------------------------------------------------------
        cluster_id, X = p6_kmeans_get_cluster_from_post(cluster_model\
        , vectorizer, df_corpus, doc_id)
        
        if verbose is True :
            print()
            print("From post ID= "+str(doc_id)+" Cluster ID= "+str(cluster_id))
        
        if cross_cluster is True :
            #-------------------------------------------------------------------------
            # Get distance between POST represented as a vector and cluster POST 
            # belongs to.
            #-------------------------------------------------------------------------
            doc_dist = p6_kmeans_dist_point(cluster_model, cluster_id, X)
            if verbose is True :
                print("ClusterID= "+str(cluster_id)\
                +" / Distance from cluster=  {0:1.2F}".format(doc_dist))
            
            #-----------------------------------------------------------------------
            # For all documents in cluster, get sorted distances between each 
            # document and cluster, reverse order.
            # Documents from cluster are extracted from corpus.
            #-------------------------------------------------------------------------
            list_dist_with_matrix \
            = p6_cluster_get_list_dist_with_matrix(cluster_model\
                                          , cluster_id, csr_matrix_corpus\
                                          , cluster_type=cluster_type)

            #-------------------------------------------------------------------
            # Get position of doc_id in list of distances of all documents in 
            # cluster. The list of distances of all documents in cluster 
            # is ordered, reversed.
            #-------------------------------------------------------------------
            pos_doc \
            = p6_cluster_get_position_distance_from_list(list_dist_with_matrix\
            , doc_dist)
            
            if verbose is True :
                print("Document position in list just after position = "\
                +str(pos_doc))
            
            
            #-------------------------------------------------------------------
            # Get list of distances in the neighborhood of document position :
            # append distances positionned before pos_doc.
            #-------------------------------------------------------------------
            
            list_sorted_selected_dist \
            = [list_dist_with_matrix[pos_doc-i] \
            for i in range(neighborhood,0,-1)]            
            
            #-------------------------------------------------------------------
            # Get list of distances in the neighborhood of document position :
            # append distances positionned after pos_doc.
            #-------------------------------------------------------------------
            try :
                list_sorted_selected_dist \
                += [list_dist_with_matrix[pos_doc+i] \
                for i in range(0,neighborhood,1) ]
            except IndexError as indexError:
                print("\n*** ERROR : Document position= "+str(pos_doc)\
                +" throwns exception= "+str(indexError)) 
                           
            if verbose is True :
                print("Number of crossed documents in cluster= "\
                +str(len(list_sorted_selected_dist)))

            #-------------------------------------------------------------------
            # In followings instructions, suggested TAG are searched across 
            # a subset of documents from cluster.
            #-------------------------------------------------------------------
            list_index = list()
            for tuple_id in list_sorted_selected_dist:
                doc_suggested_id = tuple_id[0]
                row_csr_matrix = csr_matrix_corpus[doc_suggested_id]
                
                tuple_arr_index \
                = np.where(row_csr_matrix.A[0] == np.max(row_csr_matrix.A[0]))
                arr_index = tuple_arr_index[0]
                index = arr_index[0]
                list_index.append(index)
            
            tuple_doc_index = np.where(X[0] == np.max(X[0]))     
            if verbose is True :
                print(tuple_doc_index[0], X.shape)
            list_index.append(tuple_doc_index[0])

        else : 
            #-------------------------------------------------------------------
            # Components from vectorized document are sorted, reverse order.
            # For easyness, a dataframe is used.
            #-------------------------------------------------------------------
            df = pd.DataFrame(X)
            ser_sorted \
            = df.iloc[0].sort_values(ascending=False, inplace=False)
            
            #-------------------------------------------------------------------
            # hihgest values from vector components in the vocabulary base 
            # are selected.
            #-------------------------------------------------------------------
            list_index = np.array(ser_sorted.iloc[:neighborhood].index).tolist()

    else :
        print("\n*** ERROR : cluster model= "+cluster_type+" Not supported!")
        return None
    
    #---------------------------------------------------------------------------
    # Get list fo suggested TAG as tokens in vocabulary matching with arr_index
    # previously computed.
    #---------------------------------------------------------------------------
    if verbose is True :
        print("Number of indexes from vocabulary = "+str(len(list_index)))
        print()
    list_suggested_tag = list()
    try :
        list_suggested_tag += [tag for tag, index in vectorizer.vocabulary_.items()\
            if index in list_index]    
    except ValueError as valueError:
        print("\n*** ERROR : Exception= "+str(valueError))
        print("*** ****  : list_index = "+str(list_index))
        return list(),list(),str(),str()
    
    title=df_corpus.Title[doc_id]
    post=df_corpus.Body[doc_id]
    tags=df_corpus.Tags[doc_id]
    list_tags \
    = p6_util.clean_marker_text(tags,leading_marker='<' , trailing_marker='>')
    
    return list_suggested_tag, list_tags, title, post 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_cluster_avg_accuracy(nb_test, cluster_kmean, vectorizer, csr_matrix\
                            , df_sof_test, cluster_type='kmeans'\
                            ,neighborhood=3, cross_cluster=True, verbose=False):

    range_test = range(0,nb_test,1)
    dict_suggested_tag = dict()
    #------------------------------------------------------------------------
    # For each document from test, get suggested TAG
    #------------------------------------------------------------------------
    for doc_id in range_test :
        list_suggested_tag, list_assigned_tag, title, post \
        =p6_cluster_get_suggested_tag(cluster_kmean, vectorizer, csr_matrix\
                              , df_sof_test, doc_id, cluster_type='kmeans'\
                          , neighborhood=3, cross_cluster=True, verbose=verbose)
        if(0 < len(list_suggested_tag)):
            list_intersection_sa \
            = list(set(list_suggested_tag).intersection(list_assigned_tag))
            accuracy = len(list_intersection_sa)/len(list_assigned_tag)

            dict_suggested_tag[doc_id] \
            = (list_suggested_tag,list_assigned_tag,accuracy)

    #------------------------------------------------------------------------
    # Compute average accuracy
    #------------------------------------------------------------------------
    avg_accuracy=0.0
    for tuple_value in dict_suggested_tag.values():
        avg_accuracy += tuple_value[2]
    avg_accuracy /=len(dict_suggested_tag)
    return avg_accuracy
#-------------------------------------------------------------------------------


