

void draw_tree(int im, float[][][][] result){
  sap.background(200, 200, 255);
  // given id, im
  
  int offset_x = 0;
  int offset_y = 0;
  for(int il=0; il<N_layer; il++){
    float[] sum = new float[N_neuron[il]];
    for(int j=0; j<N_neuron[il]; j++){
      if(il<N_layer-1){
        sum[j] = 0;
        for(int i=0; i<N_neuron[il+1]; i++){
          sum[j] += abs(W[il][i][j]);
        }
        
      }
    }
    int[] ind = new int[N_neuron[il]];
    float[] cpy = new float[N_neuron[il]];
    for(int i=0; i<N_neuron[il]; i++){
      ind[i] = i;
      cpy[i] = abs(sum[i]);
    }
    for(int i1=N_neuron[il]-1; i1>=0; i1--){
      for(int i2=0; i2<i1; i2++){
        if(cpy[i2]<cpy[i2+1]){
//          float tmp=cpy[i2]; cpy[i2]=cpy[i2+1]; cpy[i2+1]=tmp;
//          int tmpi=ind[i2]; ind[i2]=ind[i2+1]; ind[i2+1]=tmpi;
        }
      }
    }
    
    for(int j_=0; j_<N_neuron[il]; j_++){
      int j = ind[j_];
      PImage func = createImage(Wid, H, ARGB);
      func.loadPixels();
      for(int ix=0; ix<Wid; ix++){
        for(int iy=0; iy<H; iy++){
          func.pixels[iy*Wid+ix] = color((1.0+result[il][j][ix][iy])*128);
        }
      }
      func.updatePixels();
      
      sap.image(func, offset_x, offset_y, 40, 40);
      offset_x += 41;
      if(offset_x > 1200){
        offset_y += 41;
        offset_x = 0;
      }
      
    }
    offset_x = 0;
    offset_y += 42;
  }
}
