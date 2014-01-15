
float sigmoid(float x, float beta, int f){
//  if(x>0) println(beta*x);
//  if(x>0.2) return (0.01*(x-0.2)*beta);
//  if(x<-0.2) return (0.01*(x+0.2)*beta);
//  return 0;
//  return (x>0)?(x):0;
  float ans = (float)Math.tanh(0.5*beta*x);
//  return ans>0?ans:0;
  return ans;
//  return 1.0/(1.0+exp(-beta*x));
}

float dsigmoid(float x, float beta){
  return beta*x*(1.0-x);
}


void set_zero(int n, float[] z){
  for(int i=0; i<n; i++) z[i] = 0;
}

void set_state(int n, float[] z, float[] x){
//  println(x[0], x[1], x[2], x[3]);
  for(int i=0; i<n; i++) z[i] = x[i];
}

void set_delta(int n, float[] z, float[] x, float[] d){
  for(int i=0; i<n; i++) d[i] = x[i]-z[i];
}

void random_init(){
  for(int i=0; i<N_layer-1; i++){
    for(int j=0; j<N_neuron[i+1]; j++){
      for(int k=0; k<N_neuron[i]+1; k++){
        W[i][j][k] = random(-1, 1)/sqrt(N_neuron[i]);
      }
    }
  }
}

void forward_prop(float[] l1, float[][] w, float[] l2, int n1, int n2, int ln){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    l2[i] = 0;
    float sum = 0;
    for(int j=0; j<n1; j++){
      sum += w[i][j]*l1[j];
    }
    l2[i] = sigmoid(sum, betas[ln], 1);
  }
}

void forward_prop_for_recurrent(float[] lin, float[][] win, float[] l1, float[][] w, float[] l2, int nin, int n1, int n2, int ln){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    l2[i] = 0;
    float sum = 0;
    for(int j=0; j<n1; j++){
      sum += w[i][j]*l1[j];
    }
    for(int j=0; j<nin; j++){
      sum += win[i][j]*lin[j]*(N_layer-2-ln)/(N_layer-2);
 //     sum += win[i][j]*lin[j]*(ln==1?3:0);
    }
    l2[i] = sigmoid(sum, betas[ln], i>N_neuron[ln]/2?1:-1);
//    if(l2[i]>0) println(l2[i], ln);
  }
}

// d2->d1
void back_prop(float[] l1, float[] d1, float[][] w, float[] d2, int n1, int n2, int ln){
  for(int i=0; i<n1-1; i++){
    d1[i] = 0;
    float sum = 0;
    for(int j=0; j<n2; j++){
      sum += w[j][i]*d2[j];
    }
    d1[i] = sum * dsigmoid(l1[i], betas[ln]);
  }
}

void update_weights(float[] l1, float[][] w, float[] d2, float eta, int n1, int n2){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    for(int j=0; j<n1; j++){
//      print(w[i][j]+" ");
      w[i][j] -= eta*d2[i]*l1[j] + lambda*w[i][j];
    }
//    println();
  }
//  println();
}

void update_weights_anti_hebb(float[] l1, float[][] w, float[] l2, float eta, int n1, int n2){
  l1[n1-1] = 1.0;
  for(int i=0; i<n2; i++){
    for(int j=0; j<n1; j++){
//      print(w[i][j]+" ");
      float tmp = (l2[i])*(l1[j]);
//      if(abs(tmp)>1) tmp/=abs(tmp);
      w[i][j] -= eta*tmp + lambda*w[i][j];
    }
//    println();
  }
//  println();
}

void train_step(int ind, int lp){
  for(int i=0; i<N_layer; i++){
    set_zero(N_neuron[i], X[i]);
    set_zero(N_neuron[i], delta[i]);
  }
  
  set_state(N_neuron[0], X[0], d_in[ind]);
  for(int i=0; i<N_layer-1; i++){
    forward_prop(X[i], W[i], X[i+1], N_neuron[i]+1, N_neuron[i+1], i);
  }
  set_delta(N_neuron[N_layer-1], d_out[ind], X[N_layer-1], delta[N_layer-1]);
  for(int i=N_layer-2; i>=1; i--){
    back_prop(X[i], delta[i], W[i], delta[i+1], N_neuron[i]+1, N_neuron[i+1], i);
  }
  for(int i=0; i<N_layer-1; i++){
    update_weights(X[i], W[i], delta[i+1], eta0, N_neuron[i]+1, N_neuron[i+1]);
  }
}

void pre_train_step(int ind, int lp, int last_layer){
  int tmp = N_neuron[last_layer-1];
  N_neuron[last_layer-1] = 1;
  for(int i=0; i<last_layer; i++){
    set_zero(N_neuron[i], X[i]);
    set_zero(N_neuron[i], delta[i]);
  }
  
  set_state(N_neuron[0], X[0], d_in[ind]);
  for(int i=0; i<last_layer-1; i++){
    forward_prop(X[i], W[i], X[i+1], N_neuron[i]+1, N_neuron[i+1], i);
  }
  set_delta(N_neuron[last_layer-1], d_out[ind], X[last_layer-1], delta[last_layer-1]);
  for(int i=last_layer-2; i>=1; i--){
    back_prop(X[i], delta[i], W[i], delta[i+1], N_neuron[i]+1, N_neuron[i+1], i);
  }
  for(int i=0; i<last_layer-1; i++){
    update_weights(X[i], W[i], delta[i+1], eta0*(i+1+5)/(last_layer+5), N_neuron[i]+1, N_neuron[i+1]);
  }
  N_neuron[last_layer-1] = tmp;
}

void recurrent_step(int ind, int lp){
  for(int i=0; i<N_layer; i++){
    set_zero(N_neuron[i], X[i]);
    set_zero(N_neuron[i], delta[i]);
  }
  
  set_state(N_neuron[0], X[0], d_in[ind]);
  for(int i=1; i<N_layer-2; i++){
    forward_prop_for_recurrent(X[0], W[0], X[i], W[1], X[i+1], N_neuron[0], N_neuron[i]+1, N_neuron[i+1], i);
  }
  forward_prop_for_recurrent(X[0], W[0], X[N_layer-2], W[N_layer-2], X[N_layer-1], N_neuron[0], N_neuron[N_layer-2]+1, N_neuron[N_layer-1], N_layer-2);
  
  set_delta(N_neuron[N_layer-1], d_out[ind], X[N_layer-1], delta[N_layer-1]);
//  back_prop(X[N_layer-2], delta[N_layer-2], W[N_layer-2], delta[N_layer-2+1], N_neuron[N_layer-2]+1, N_neuron[N_layer-2+1], N_layer-2);
  update_weights_anti_hebb(X[N_layer-2], W[N_layer-2], delta[N_layer-1], 0.1*eta0, N_neuron[N_layer-2]+1, N_neuron[N_layer-1]);
//  update_weights_anti_hebb(X[N_layer-3], W[1], delta[N_layer-2], -0.2*eta0, N_neuron[N_layer-3]+1, N_neuron[N_layer-2]);
  
//    update_weights_anti_hebb(X[1], W[1], X[2], 0.3*eta0/(lp+1), N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[0], W[0], X[2], 0.01*eta0/(lp+1), N_neuron[0]+1, N_neuron[1]);
//    update_weights_anti_hebb(X[0], W[0], X[8], -0.01*eta0/(lp+1), N_neuron[0]+1, N_neuron[1]);
//    update_weights_anti_hebb(X[1], W[1], X[2], 0.01*eta0/(lp+1), N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[2], W[1], X[3], 0.01*eta0/(lp+1), N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[3], W[1], X[4], 0.01*eta0/(lp+1), N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[4], W[1], X[5], 0.01*eta0/(lp+1), N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[3], W[1], X[4], eta0, N_neuron[1]+1, N_neuron[2]);
//    update_weights_anti_hebb(X[5], W[1], X[6], 0.01*eta0/(lp+1), N_neuron[4]+1, N_neuron[5]);
//    update_weights_anti_hebb(X[6], W[1], X[7], 0.01*eta0/(lp+1), N_neuron[4]+1, N_neuron[5]);
//    update_weights_anti_hebb(X[7], W[1], X[8], -0.01*eta0, N_neuron[4]+1, N_neuron[5]);
//    update_weights_anti_hebb(X[8], W[1], X[9], -0.01*eta0, N_neuron[4]+1, N_neuron[5]);
//  for(int i=0; i<N_layer-2; i++){
//    update_weights_anti_hebb(X[i], W[1], X[i+1], eta0, N_neuron[i]+1, N_neuron[i+1]);
//  }
  /*
  set_delta(N_neuron[N_layer-1], d_out[ind], X[N_layer-1], delta[N_layer-1]);
  for(int i=N_layer-2; i>=1; i--){
    back_prop(X[i], delta[i], W[i], delta[i+1], N_neuron[i]+1, N_neuron[i+1], i);
  }
  for(int i=0; i<N_layer-1; i++){
    update_weights(X[i], W[i], delta[i+1], eta0, N_neuron[i]+1, N_neuron[i+1]);
  }
  */
}
