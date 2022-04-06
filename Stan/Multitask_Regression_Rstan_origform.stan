functions {
  real customProb(int n, int p, int r, real q, real tow02, real tow2, real a, real b, 
                  matrix Y, matrix X, matrix L, matrix S_U, matrix S_V, vector d, 
                  matrix S, real sigma2){
    matrix[n,p] Z;
    real term1;
    real term2;
    real term3;
    real out;
    
    Z = Y - (X*(L + S));
    
    term1 = - (((n*p*0.5) + a + 1)*log(sigma2)) - ((1/(2*sigma2))*trace(crossprod(Z)));
    term2 = - ((0.5)*trace(S_U + S_V)) - ((0.5)*(d'*d)) - (b/sigma2);
    term3 = sum(log((q/sqrt(tow02))*exp((-1/(2*tow02))*(S .* S)) + ((1-q)/sqrt(tow2))*exp((-1/(2*tow2))*(S .* S))));
    
    out = term1 + term2 + term3;
    return(out);
  }
}

data {
  int<lower=0> n;
  int<lower=0> p;
  int<lower=1> r;
  real<lower=0,upper=1> q;
  real<lower=0> tow02;      // tow_0^2
  real<lower=0> tow2;       // tow^2
  real<lower=0> a;          // prior of sigma2 ~ Inv_gamma(a,b)
  real<lower=0> b;
  matrix[n,p] Y;
  matrix[n,p] X;
}

parameters {
  matrix[n,r] X_1;
  matrix[p,r] X_2;
  vector[r] d;
  matrix[n,p] S;
  real<lower=0> sigma2;
}

transformed parameters {
  // We calculate the orthogonal matrix U and V from X_1 and X_2 respectively using
  // the polar decomposition.
  matrix[n,r] U;
  matrix[p,r] V;
  matrix[r,r] S_U;
  matrix[r,r] S_V;
  matrix[n,p] L;
  {
    vector[r] eval1;
    vector[r] eval2;
    vector[r] eval_trans1;
    vector[r] eval_trans2;
    vector[r] d1;
    vector[r] d2;
    matrix[r,r] evec1;
    matrix[r,r] evec2;
    eval1 = eigenvalues_sym(crossprod(X_1));   // same as (X_1)'*(X_1)
    eval2 = eigenvalues_sym(crossprod(X_2));
    for(l in 1:r){
      eval_trans1[l] = 1/sqrt(eval1[l]);
      eval_trans2[l] = 1/sqrt(eval2[l]);
      d1[l] = d[r+1-l];
    }
    evec1 = eigenvectors_sym(crossprod(X_1));
    evec2 = eigenvectors_sym(crossprod(X_2));
    U = (X_1)*diag_post_multiply(evec1, eval_trans1)*evec1';
    // Use diag_post_multiply(evec, eval_trans), which is more efficient than using
    // diag_matrix(). Similarly true for pre-multiplication as well.
    V = (X_2)*diag_post_multiply(evec2, eval_trans2)*evec2';
    
    S_U = crossprod(X_1);  // same as (X_1)'*(X_1)
    S_V = crossprod(X_2);
    
    d2 = sort_desc(exp(cumulative_sum(log(d1))));
    L = diag_post_multiply(U,d2)*V';             // D = diag_matrix(d2);  L = U*D*V'
  }
}

model {
  // Here we specify the log of the probability density the joint posterior.
  target += customProb(n, p, r, q, tow02, tow2, a, b, Y, X, L, S_U, S_V, d, S, sigma2);
}
