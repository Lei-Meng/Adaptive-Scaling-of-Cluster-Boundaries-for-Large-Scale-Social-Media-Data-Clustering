
import scipy.io as sio
import numpy as np


def am_art(M,label,rho,save_path_root):
    '''
    % M: numpy arrary; m*n feature matrix; m is number of objects and n is number of visual features
    %rho: the vigilance parameter
    %save_path_root: path to save clustering results for further analysis
    '''

    NAME = 'am_art'
#-----------------------------------------------------------------------------------------------------------------------
# Input parameters
    alpha = 0.01 # no need to tune; used in choice function; to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime); give priority to choosing denser clusters
    beta = 0.6 # has no significant impact on performance with a moderate value of [0.4,0.7]
    sigma = 0.000005 #the percentage to enlarge or shrink vigilance region

    # rho needs carefully tune; used to shape the inter-cluster similarity; rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    #rho = 0.6



# -----------------------------------------------------------------------------------------------------------------------
# Initialization

    #complement coding
    M = np.concatenate([M,1-M], 1)

    #get data sizes
    row, col = M.shape




# -----------------------------------------------------------------------------------------------------------------------
# Clustering process

    print( NAME + "algorithm starts")

    #create initial cluster with the first data sample
        #initialize cluster parameters
    Wv = np.zeros((row, col))
    J = 0  # number of clusters
    L = np.zeros((1,row), dtype=np.int)  # size of clusters; note we set to the maximun number of cluster, i.e. number of rows
    Assign = np.zeros((1,row), dtype=np.int)  # the cluster assignment of objects
    rho_0 = rho * np.ones((1,row))
    T_values = np.zeros((1,row))-2 #store the index of clusters incur reset - for computing new rhos

        #first cluster
    print('Processing data sample 0')
    Wv[0, :] = M[0, :]
    J = 1
    L[0,J-1] = 1
    Assign[0,0] = J-1 #note that python array index trickily starts from 0

    #processing other objects
    for n in range(1,row):

        print('Processing data sample %d' % n)

        T_max = -1 #the maximun choice value
        winner = -1 #index of the winner cluster

        #compute the similarity with all clusters; find the best-matching cluster
        for j in range(0,J):

            #compute the match function
            Mj_numerator_V = np.sum(np.minimum(M[n,:],Wv[j,:]))
            Mj_V = Mj_numerator_V / np.sum(M[n,:])

            # compute choice function
            T_values[0,j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

            if Mj_V >= rho_0[0,j] and T_values[0,j] >= T_max:
                T_max = T_values[0,j]
                winner = j

        #update rho of reset winner clusters
        a = np.where(T_values[0, :] >= T_max)
        if winner > -1:
            b = a[0]
            a = np.delete(b,np.where(b==winner))
        rho_0[0,a] = (1-sigma) * rho_0[0,a]


        #Cluster assignment process
        if winner == -1: #indicates no cluster passes the vigilance parameter - the rho
            #create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            L[0, J - 1] = 1
            Assign[0,n] = J - 1
        else: #if winner is found, do cluster assignment and update cluster weights
            #update cluster weights
            Wv[winner, :] = beta * np.minimum(Wv[winner, :], M[n, :]) + (1 - beta) * Wv[winner, :]
            #cluster assignment
            L[0, winner] += 1
            Assign[0,n] = winner
            rho_0[0,winner] = (1+sigma) * rho_0[0,winner]



    print("algorithm ends")
    # Clean indexing data
    Wv = Wv[0: J, :]
    L = L[:, 0: J]

# -----------------------------------------------------------------------------------------------------------------------
# performance calculation

    # confusion-like matrix
    number_of_class = int(max(label)) +1
    confu_matrix = np.zeros((J,number_of_class))

    for i in range(0 ,row):
        confu_matrix[Assign[0,i] ,int(label[i])] += 1

    # compute dominator class and its size in each cluster
    max_value = np.amax(confu_matrix ,axis=1)
    max_index = np.argmax(confu_matrix ,axis=1)
    size_of_classes = np.sum(confu_matrix ,axis=0)

    # compute precision, recall
    precision = max_value / L[0 ,:]

    recall = np.zeros((J))
    for i in range(0,J):
        recall[i] = max_value[i] / size_of_classes[max_index[i]]


    #intra_cluster distance - Euclidean
    intra_cluster_distance = np.zeros((J))
    for i in range(0,row):
        temp1 = np.sqrt(np.sum(np.square(Wv[Assign[0,i],0:(col//2)] - M[i,0:(col//2)])))
        temp2 = np.sqrt(np.sum(np.square(1 - Wv[Assign[0, i], (col // 2):] - M[i, 0:(col // 2)])))

        intra_cluster_distance[Assign[0,i]] += (temp1 + temp2) / 2 #compute average distance between bottom-left and upper-right points of the cluster and the input pattern

    intra_cluster_distance = intra_cluster_distance[:] / L[0,:]

    # inter_cluster distance - Euclidean
    inter_cluster_distance = np.zeros(((J*(J-1))//2))
    len = 0
    for i in range(0,J):
        for j in range(i+1,J):
            temp = np.square(Wv[i,:] - Wv[j,:])
            inter_cluster_distance[len] = (np.sqrt(np.sum(temp[0:(col//2)])) + np.sqrt(np.sum(temp[(col//2):]))) / 2 #compute the average distance between bottom-left and upper-right points of two clusters
            len += 1

# -----------------------------------------------------------------------------------------------------------------------
# save results

    sio.savemat(save_path_root + str(number_of_class) + '_class_' + NAME + '_rho_' + str(rho) + '.mat',
                {'precision': precision, 'recall': recall, 'cluster_size': L,
                 'intra_cluster_distance': intra_cluster_distance, 'inter_cluster_distance': inter_cluster_distance,
                 'Wv': Wv, 'confu_matrix': confu_matrix})


    return 0

