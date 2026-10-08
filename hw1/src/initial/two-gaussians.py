import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html



# Set random seed for reproducibility
np.random.seed(42)



def get_mean_and_covariance(X):
    """ Get the mean  mu in IR^d and covariance Sigma in IR^{d times d} from samples in matrix X in IR^{n x d}. Each row corresponds to a different sample"""
    e_mean = np.mean(X,0)
    e_cov = np.cov(X,rowvar = False)
    return (e_mean,e_cov)



def tau_range(X,g_pos,g_neg):
    """return the range of taus that one needs to span to create the ROC curve with these discriminant functions"""
    n,d = X.shape
    scores = [g_pos(X[i,:])-g_neg(X[i,:]) for i in range(n)]
    return min(scores),max(scores)
   

def predict(X,g_pos,g_neg,tau):
    """Generate predictions (+1/-1) over samples in X using discriminant functions g_pos, g_neg with threshold tau"""
    y_pred = []
    n,d = X.shape
    for i in range(n):
        x = X[i,:]
        y_pred.append(np.sign(g_pos(x)-g_neg(x)-tau))
    return y_pred     
    


def metrics(y_pred,y_true):
    """ Output the FPR and the TPR. These are defined in terms of the following quantities:
      
                 P = # entries y_true in that are  + 1
                 N = # entries y_true in that are  - 1
 
                 
                 TP = # entries for which  y_pred=+1 and y_true=+1  
                 FP = # entries for which  y_pred=-1 but y_true = +1  
                 TN = # entries for which y_pred=-1 and y_true = -1  
                 FN =  # entries for which y_pred=-1 but y_true = +1  


                 TPR = TP / P
                 FPR = FP / N
        Inputs are:
             - y_pred: predictions
             - y_true: ground truth values

        The return value is a tuple containing
             - #P,#N,#TP,#FP,#TN,#FN
    """

    pairs = [  (int(x),int(y))  for (x,y) in zip(y_pred,y_true)]
    new_pairs = [ (pred_label,pred_label*true_label)  for (pred_label,true_label) in pairs ]        
    

    TP = 1.*new_pairs.count( (1,1) )
    FP = 1.*new_pairs.count( (1,-1) )
    TN = 1.*new_pairs.count( (-1,1) )
    FN = 1.*new_pairs.count( (-1,-1) )
    P = TP+FN
    N = TN+FP

    TPR = 1.*TP/P
    FPR = 1.*FP/N
    return FPR,TPR

 
def naive_classifier(q,n):
    return -np.sign(np.random.rand(n)-q)


p = 0.5 # prior probability of positive class
n = 1000 # total number of samples


if __name__=="__main__":
    # Generate two multivariate Gaussian random variable objects
    mean_pos, cov_pos = [1, 3], [[1, 0.8], [0.8, 1]]
    e=eigvals(cov_pos)
    print("Eigenvalues of cov_pos are:",e)
    rv_pos = multivariate_normal(mean_pos, cov_pos)

    mean_neg, cov_neg = [4, 4], [[0.99, 0.81], [0.81, 1.01]]
    e=eigvals(cov_neg)
    print("Eigenvalues of cov_neg are:",e)
    rv_neg = multivariate_normal(mean_neg, cov_neg)


    prior_pos = p
    prior_neg = 1-p

    # Generate samples
    n_pos = np.random.binomial(n,p)
    n_neg = n - n_pos
    X_pos = rv_pos.rvs(size = n_pos)
    X_neg = rv_neg.rvs(size = n_neg)


    # Create a scatter plot for the samples
    fig	 = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_pos[:, 0], X_pos[:, 1], label='Sea Bass (+)', alpha=0.7,c="blue")
    ax.scatter(X_neg[:, 0], X_neg[:, 1], label='Salmon (-)', alpha=0.7, c="orange")

    # Add contours of the pdfs
    x1, x2 = np.mgrid[-1:8:.01, -1:6:.01]
    positions = np.dstack((x1, x2))
    ax.contour(x1, x2, rv_pos.pdf(positions),levels=5,colors="blue")
    ax.contour(x1, x2, rv_neg.pdf(positions), levels=5, colors="orange")


    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()


    plt.savefig("two-gaussians.pdf", format="pdf", bbox_inches="tight")

   
    # Estimate the mean, covariance, and prior for each Gaussian from samples
    e_mean_pos, e_cov_pos = get_mean_and_covariance(X_pos)
    e_mean_neg, e_cov_neg = get_mean_and_covariance(X_neg)
    e_prior_pos = 1. * n_pos / n
    e_prior_neg = 1. * n_neg / n

    print ("Positive class estimates:", e_mean_pos,e_cov_pos,e_prior_pos)
    print ("Negative class estimates:", e_mean_neg,e_cov_neg,e_prior_neg)


    # Join samples to create a classification dataset
    X = np.concatenate((X_pos,X_neg))
    y = np.concatenate( (np.ones(n_pos),-np.ones(n_neg)) )
 
    
    #create ROC Curve for naive classifier


    FPR_naive_list=[]
    TPR_naive_list=[]
    delta = (1.0-0.0)/10
    print('Scanning tau for naive...')
    for q in np.arange(0.,1+delta,delta):
        #generate predictions
        y_pred = naive_classifier(q,n)

        #compute FPR and TPR
        FPR,TPR = metrics(y_pred,y)
        print('FPR =',FPR,'TPR =',TPR)
        FPR_naive_list.append(FPR)
        TPR_naive_list.append(TPR)
    print('...done with',len(FPR_naive_list),'measurements')
    
    # Plot the ROC curve
    fig	 = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(FPR_naive_list, TPR_naive_list, linestyle='dashed' ,color='yellow',label = 'Naive')

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.legend(loc='lower right')


    plt.savefig("ROC.pdf", format="pdf", bbox_inches="tight")

    
    
