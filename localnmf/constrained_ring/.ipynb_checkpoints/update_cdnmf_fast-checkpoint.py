def update_cdnmf_fast(W, HHt, XHt, permutation):
   
        
    n_components = W.shape[1]
    n_samples = W.shape[0]  # n_   
    violation = 0
    for s in range(n_components):
        t = permutation[s]
        for i in range(n_samples):
            grad = -XHt[i, t]
            
            for r in range(n_components):
                grad += HHt[t, r] * W[i, r]
                
            pg = min(0., grad) if W[i, t] == 0 else grad
            
            violation = 0

            # Hessian
            hess = HHt[t, t]

            if hess != 0:
                W[i, t] = W[i,t] - grad / hess
                
                if W[i,t] < 0:
                    W[i,t] = 0
                
                elif W[i,t] > 5./n_components:
                   
                    W[i,t] = 5./n_components
                   
                else:
                    W[i, t] = W[i,t]
               
       
    return violation