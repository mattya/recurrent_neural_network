//float beta = 5.0;
float lambda = 0.000000000;
float eta0 = 0.05;

int N_images = 2;
PImage[] imgs;

int Wid = 60;
int H = 60;
int NTrain = 500;
int NMax = N_images*Wid*H;

int Loop = 50;

int N_layer = 14;
//int[] N_neuron = {2+N_images, 6,6,6,6,6,6,6,6,6};
int[] N_neuron = {2+N_images, 20,20,20,20,20,20,20,20,20,20,20,20,1};
//float[] betas = {1, 3, 5, 7, 9, 7, 5, 3, 1};
//float[] betas = {1,2,3,4,5,6,7,8,9};
//float[] betas = {0.5,1,1.5,2,2.5,3,3.5,4,4.5};
//float[] betas = {2,2,2,2,3,4,5,6,7};
//float[] betas = {1,3,5,7,9,11,13,15,17};
//float[] betas = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
//float[] betas = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
float[] betas = {15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15};
//float[] betas = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//float[] betas = {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
int NNMax = 123;

float[][] d_in, d_out;

float[][] X;
float[][][] W;
float[][] delta;

void alloc(){
  d_in = new float[NMax][2+N_images];
  d_out = new float[NMax][N_neuron[N_layer-1]];
  
  X = new float[N_layer][NNMax+1];
  delta = new float[N_layer][NNMax+1];
  W = new float[N_layer][NNMax+1][NNMax+1];
  println("alloc_done");
}

int cur_last_layer = 0;
void train(){
//  random_init();
  for(int i=0; i<Loop; i++){
//    println(i);
    for(int j=0; j<NTrain; j++){
      if(random(0, 1)<1.5){
//        train_step((int)random(0, NMax), step);
//        pre_train_step((int)random(0, NMax), step, cur_last_layer);
        recurrent_step((int)random(0, NMax), step);
      }
    }
    
  }
  /*
    if(step%1==0){
      for(int j=0; j<N_layer-1; j++){
        String[] lines = new String[N_neuron[j+1]];
        for(int k=0; k<N_neuron[j+1]; k++){
          lines[k] = "";
          for(int l=0; l<N_neuron[j]+1; l++){
            lines[k] += W[j][k][l] + " ";
          }
        }
        saveStrings("e_"+step+"_"+j+".txt", lines);
      }
      
    }
    */
}
void calc_all(int im, float[][][][] result){
  for(int ix=0; ix<Wid; ix+=1){
    for(int iy=0; iy<H; iy+=1){
      for(int il=0; il<N_layer; il++) set_zero(N_neuron[il], X[il]);
      X[0][0] = map(ix, 0, Wid, -1, 1);
      X[0][1] = map(iy, 0, H, -1, 1);
      for(int ik=0; ik<N_images; ik++){
        X[0][2+ik] = (ik==im?1:-1);
      }
      for(int il=0; il<N_layer-1; il++){
        forward_prop(X[il], W[il], X[il+1], N_neuron[il]+1, N_neuron[il+1], il);
      }
      
      for(int il=0; il<N_layer; il++){
        for(int j=0; j<N_neuron[il]; j++){
          result[il][j][ix][iy] = X[il][j];
        }
      }
    }         
  }
}

void calc_all_recurrent(int im, float[][][][] result){
  for(int ix=0; ix<Wid; ix+=1){
    for(int iy=0; iy<H; iy+=1){
      for(int il=0; il<N_layer; il++) set_zero(N_neuron[il], X[il]);
      set_state(N_neuron[0], X[0], d_in[im*Wid*H + iy*Wid+ix]);
      
      for(int i=1; i<N_layer-2; i++){
        forward_prop_for_recurrent(X[0], W[0], X[i], W[1], X[i+1], N_neuron[0], N_neuron[i]+1, N_neuron[i+1], i);
      }
  forward_prop_for_recurrent(X[0], W[0], X[N_layer-2], W[N_layer-2], X[N_layer-1], N_neuron[0], N_neuron[N_layer-2]+1, N_neuron[N_layer-1], N_layer-2);
      
      for(int il=0; il<N_layer; il++){
        for(int j=0; j<N_neuron[il]; j++){
          result[il][j][ix][iy] = X[il][j];
        }
      }
      
    }         
  }
}

float calc_error(int im, float[][][][] result){
  float ret = 0;
  for(int ix=0; ix<Wid; ix+=1){
    for(int iy=0; iy<H; iy+=1){
      ret += sq(brightness(imgs[im].get(ix, iy)) - 256*result[N_layer-1][0][ix][iy]);
    }
  }
  return ret/(Wid*H);
}

void disp_func(int im, float[][][][] result){
  PImage func = createImage(Wid, H, RGB);
  func.loadPixels();
  for(int i=0; i<func.width; i+=1){
    for(int j=0; j<func.height; j+=1){
      color c = color(128*(1.0+result[N_layer-1][0][i][j]));
      for(int ix=0; ix<1; ix++) for(int iy=0; iy<1; iy++)
        func.pixels[(j+iy)*func.width+i+ix] = c;
    }
  }
  func.updatePixels();
  
  image(imgs[im], Wid*(im%30), H*2*(im/30));
  image(func, Wid*(im%30), H*2*(im/30)+H);
  stroke(0);
  line(Wid*im, 0, Wid*im, H);
  line(0, height/2, width, height/2);
    
}


void img_to_d(){
  for(int im=0; im<N_images; im++){
    imgs[im].loadPixels();
    for(int i=0; i<H; i++){
      for(int j=0; j<Wid; j++){
        d_in[im*Wid*H + j*Wid+i][0] = map(i, 0, Wid, -1, 1);
        d_in[im*Wid*H + j*Wid+i][1] = map(j, 0, H, -1, 1);
        for(int k=0; k<N_images; k++){
          d_in[im*Wid*H + j*Wid+i][2+k] = (im==k?1:-1);
        }
        d_out[im*Wid*H + j*Wid+i][0] = map(red(imgs[im].pixels[j*Wid+i]), 0, 255, -1, 1);
//        d_out[im*Wid*H + j*Wid+i][1] = map(green(imgs[im].pixels[j*Wid+i]), 0, 255, 0, 1);
//        d_out[im*Wid*H + j*Wid+i][2] = map(blue(imgs[im].pixels[j*Wid+i]), 0, 255, 0, 1);
      }
    }
  }
}

