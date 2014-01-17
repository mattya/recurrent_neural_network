import processing.core.*;

public class Weight {
  PApplet pa;
  int N, M;
  float[][] mat;
  
  int from, to;
  int dir;  // 0: undirected, 1: directed
  int type; // 0: fixed, 1: RBM
  float alpha0, alpha1;
  
//  Weight(int N0, int M0){
//    
//  }
  Weight(int N0, int M0, int f0, int t0, int d0, int type0, float a0, float a1, PApplet p0){
    pa = p0;
    N = N0;
    M = M0;
    from = f0;
    to = t0;
    dir = d0;
    type = type0;
    alpha0=a0;
    alpha1=a1;
    mat = new float[M][N];
    random_init();
  }
//  Weight(Weight w0){
//    
//  }
  
  void random_init(){
    for(int i=0; i<M; i++){
      float sum = 0;
      for(int j=0; j<N; j++){
        if(pa.random(0, 1)<0.8) mat[i][j] = 0;
        else mat[i][j] = pa.random(-1.0f, 1.0f)*10f;
      }
    }
  }
  
  float infinity_norm(){
    float mx = 0;
    for(int i=0; i<M; i++){
      float sum = 0;
      for(int j=0; j<N; j++){
        sum += Math.abs(mat[i][j]);
      }
      if(sum>mx) mx = sum;
    }
    return mx;
  }
}
