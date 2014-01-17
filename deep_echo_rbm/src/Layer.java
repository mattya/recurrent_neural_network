import processing.core.*;
public class Layer {
  PApplet pa;
  int id;
  String name;
  
  int N_neurons;
  int N_targets;
  Weight[] ws;
  
  float beta;
  float tau;
  float[] x;   // x
  float[] nx;  // next_x
  float[] b;   // bias
  
  int tmp;
  
  Layer(int id0, String name0, int N_neurons0, int N_targets0, float tau0, PApplet p0){
    pa = p0;
    id = id0; 
    name = name0;
    N_neurons = N_neurons0;
    N_targets = N_targets0;
    beta = Global.beta;
    tau = Global.tau;
    ws = new Weight[N_targets];
    x = new float[N_neurons];
    nx = new float[N_neurons];
    b = new float[N_neurons];
    tau = tau0;
    random_init();
  }
  
  void random_init(){
    for(int i=0; i<N_neurons; i++){
      b[i] = pa.random(-0.10f, 0.10f);
    }
  }
  
  float nonlinear(float x, float beta){
    float ans = (float)Math.tanh(beta*x);
    return ans;
  }
  
  void append_weight(Weight w){
    ws[tmp] = w;
    tmp++;
  }

  void set_x(float[] x0) {
    for(int i=0; i<N_neurons; i++){
      x[i] = x0[i];
    }
  }
  
  void set_zero_x(){
    for(int i=0; i<N_neurons; i++){
      x[i] = -1;
    }
  }
  
  void set_zero_nx(){
    for(int i=0; i<N_neurons; i++){
      nx[i] = 0;
    }
  }
  
  void apply_bias_and_nonlinear(int f){
//    if(tau>0){
      for(int i=0; i<N_neurons; i++){
        if(f==-1) nx[i] = pa.random(-1.05f, 1.05f)<nonlinear(nx[i]+b[i], beta)?1:-1;
        else nx[i] = pa.random(-1.05f, 1.05f)<(tau*x[i]+(nonlinear(nx[i]+b[i], beta)))/(tau+1.0f)?1:-1;
        //nx[i] = (tau*x[i]+nonlinear(nx[i]+b[i], beta))/(tau+1.0f);
      }
//    }
  }
  
  void update(){
//    if(tau>0){
      for(int i=0; i<N_neurons; i++){
//        pa.println(name, x[i], nx[i]);
//        x[i] = pa.random(-1, 1)<nx[i]?1:-1;
        x[i] = nx[i];
      }
//    }
  }
  
  PImage visualize(){
    int wid = Global.wid;
    PImage img = pa.createImage(wid, (wid+1)*(N_neurons), pa.ARGB);
    img.loadPixels();
    for(int i=0; i<N_neurons; i++){
      for(int ix=0; ix<wid; ix++){
        for(int iy=0; iy<wid; iy++){
          int id = i*wid*(wid+1) + iy*wid + ix;
          img.pixels[id] = pa.color(pa.map(x[i], -1, 1, 0, 255));
        }
      }
    }
    img.updatePixels();
    return img;
  }

}
