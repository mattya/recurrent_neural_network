import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import processing.core.*;

public class Network {
  PApplet pa;
  int N_layers;
  Layer[] layers;
  
  int N_weights;
  Weight[] weights;
  
  int N_inputs;
  int[] input_layer_id;
  
  /**
   *  input sample
   *  3 2 1            (N_layers, N_weights, N_inputs)
   *  0 l_in 28 1      (layer_id, layer_name, N_neurons, N_targets)
   *  1 l_hid0 100 2
   *  2 l_hid1 50 1
   *  0 1 0 1          (from, to, dir, type)
   *  1 2 0 1
   *  0                (input_layer_id)
  */
  Network(String[] arc, PApplet p0){
    pa = p0;
    String[] spl;
    spl = pa.split(arc[0], ' ');
    N_layers = Integer.parseInt(spl[0]);
    N_weights = Integer.parseInt(spl[1]);
    N_inputs = Integer.parseInt(spl[2]);
    layers = new Layer[N_layers];
    weights = new Weight[N_weights];
    input_layer_id = new int[N_inputs];
    
    int offset = 1;
    for(int i=0; i<N_layers; i++){
      spl = pa.split(arc[offset+i], ' ');
      layers[i] = new Layer(Integer.parseInt(spl[0]), spl[1], Integer.parseInt(spl[2]), Integer.parseInt(spl[3]), Float.parseFloat(spl[4]), pa);
    }
    offset = 1+N_layers;
    for(int i=0; i<N_weights; i++){
      spl = pa.split(arc[offset+i], ' ');
      int from = Integer.parseInt(spl[0]);
      int to = Integer.parseInt(spl[1]);
      weights[i] = new Weight(layers[from].N_neurons, layers[to].N_neurons, from, to, Integer.parseInt(spl[2]), Integer.parseInt(spl[3]), Float.parseFloat(spl[4]), Float.parseFloat(spl[5]), pa);
      
      layers[from].append_weight(weights[i]);
      if(weights[i].dir==0){
        layers[to].append_weight(weights[i]);
      }
    }
    offset = 1+N_layers+N_weights;
    spl = pa.split(arc[offset], ' ');
    for(int i=0; i<N_inputs; i++){
      input_layer_id[i] = Integer.parseInt(spl[i]);
//      layers[input_layer_id[i]].tau = -1;
    }
    
  }
  
  void prop_impl(float alpha, float[] l1, float[][] w, float[] l2, int n1, int n2, int transpose){
    for(int i=0; i<n2; i++){
      float sum = 0;
      if(transpose==0){
        for(int j=0; j<n1; j++){
          sum += alpha*w[i][j]*l1[j];
        }
        sum /= Math.sqrt(n1);
      }else{
        for(int j=0; j<n1; j++){
          sum += alpha*w[j][i]*l1[j];
        }
        sum /= Math.sqrt(n1);
      }
//      pa.println(i, n1, n2, l2[i], sum);
      l2[i] += sum;
    }
  }
  
  void prop(Weight w, int dir){
//    pa.println("prop:", w.from, w.to);
    if(dir==0) prop_impl(w.alpha0, layers[w.from].x, w.mat, layers[w.to].nx, layers[w.from].N_neurons, layers[w.to].N_neurons, 0);
    else{
      prop_impl(w.alpha1, layers[w.to].x, w.mat, layers[w.from].nx, layers[w.to].N_neurons, layers[w.from].N_neurons, 1);
    }
  }
  
  void update_weights_impl(float[] l1, float[][] w, float[] l2, int n1, int n2, float eta){
    for(int i=0; i<n2; i++){
      for(int j=0; j<n1; j++){
        float tmp = (l2[i])*(l1[j]);
        w[i][j] += eta*tmp;
      }
    }
  }
  
  // 0: hebb
  // 1: anti-hebb
  void update_weights(Weight w, int rule){
    float eta = Global.eta;
    if(rule==1) eta = -1*eta;
    update_weights_impl(layers[w.from].x, w.mat, layers[w.to].x, layers[w.from].N_neurons, layers[w.to].N_neurons, eta);
  }
  
  void update_bias(Layer l, int rule){
    float eta = Global.eta;
    if(rule==1) eta = -1.0f*eta;
    for(int i=0; i<l.N_neurons; i++){
      l.b[i] += eta*l.nx[i];
    }
  }
  
  /**
   * @param kind 
   * -1: none
   *  0: train
   *  1: test
   * @param laerning_type
   * -1: none
   *  0: hebb
   *  1: anti-hebb
   */
  void one_step(int data_kind, int data_id, int learning_type, int state){
    // set_zero
    for(int i=0; i<N_layers; i++){
      layers[i].set_zero_nx();
    }
    
    // set input
    /*
    for(int i=0; i<N_inputs; i++){
      int il = input_layer_id[i];
//      if(data_kind==-1) layers[il].set_zero_x();
      if(data_kind==-1);
      else layers[il].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, i, data_id));
    }
    */
    if(state%2==0){
//      layers[input_layer_id[0]].set_zero_x();
//      layers[input_layer_id[1]].set_zero_x();
      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    if(state==1){
      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
//      layers[input_layer_id[1]].set_zero_x();
    }
    if(state==3){
//      layers[input_layer_id[0]].set_zero_x();
      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    if(state==5){
    layers[input_layer_id[0]].set_zero_x();
    layers[input_layer_id[1]].set_zero_x();
//      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
//      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    
    // prop
    for(int i=0; i<N_weights; i++){
      prop(weights[i], 0);
      if(weights[i].dir==0) prop(weights[i], 1);
    }
    for(int i=0; i<N_layers; i++){
      layers[i].apply_bias_and_nonlinear(0);
    }
    
    // update_weights
    /*
    if(state%2==1){
      update_weights(weights[0], 0);
      update_weights(weights[4], 0);
      update_bias(layers[0], 0);
      update_bias(layers[2], 0);
    }else{
      update_weights(weights[0], 1);
      update_weights(weights[4], 1);
      update_bias(layers[0], 1);
      update_bias(layers[2], 1);
    }
    if(state/2==1){
      update_weights(weights[1], 0);
      update_weights(weights[5], 0);
      update_bias(layers[1], 0);
      update_bias(layers[3], 0);
    }else{
      update_weights(weights[1], 1);
      update_weights(weights[5], 1);
      update_bias(layers[1], 1);
      update_bias(layers[3], 1);
    }
    */
    /*
    for(int i=0; i<N_weights; i++){
      if(weights[i].type==1){
        if(state%2==0) update_weights(weights[i], 1);
        else update_weights(weights[i], 0);
      }else{
        update_weights(weights[i], 2);
      }
    }
    for(int i=0; i<N_layers; i++){
      if(state%2==0) update_bias(layers[i], 1);
      else update_bias(layers[i], 0);
    }
    */
    /*
    if(learning_type>=0){
      for(int i=0; i<N_weights; i++){
        if(weights[i].type==1)
          update_weights(weights[i], learning_type);
      }
    }
    */
    // contrastive divergence 1
    if(state%1==0){
      for(int i=0; i<N_weights; i++){
        if(weights[i].type==1){
          update_weights(weights[i], 0);
        }
      }
      for(int i=0; i<N_layers; i++){
        update_bias(layers[i], 0);
      }
    }
    else if(state==5){
      
      for(int i=0; i<N_weights; i++){
        if(weights[i].type==1){
 //         update_weights(weights[i], 1);
        }
      }
      for(int i=0; i<N_layers; i++){
 //       update_bias(layers[i], 1);
      }
      
    }
    
    for(int i=0; i<N_layers; i++){
      layers[i].update();
    }

    // contrastive divergence 2
    if(state%1==0){
      for(int i=0; i<N_layers; i++){
        layers[i].set_zero_nx();
      }
      for(int i=0; i<N_weights; i++){
        prop(weights[i], 0);
        if(weights[i].dir==0) prop(weights[i], 1);
      }
      for(int i=0; i<N_layers; i++){
        layers[i].apply_bias_and_nonlinear(-1);
      }
      for(int i=0; i<N_weights; i++){
        if(weights[i].type==1){
          Weight w = weights[i];
          update_weights_impl(layers[w.from].nx, w.mat, layers[w.to].nx, layers[w.from].N_neurons, layers[w.to].N_neurons, -1f*Global.eta);
        }
      }
      for(int i=0; i<N_layers; i++){
        update_bias(layers[i], 1);
      }
    }
    
    
    if(state==1){
      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
//      layers[input_layer_id[1]].set_zero_x();
    }
    else if(state==3){
//      layers[input_layer_id[0]].set_zero_x();
      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    else if(state==5){
//      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
//      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }else{
      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    
    /*
    if(state%2==1){
      layers[input_layer_id[0]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 0, data_id));
    }
    if(state/2==1){
      layers[input_layer_id[1]].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, 1, data_id));
    }
    */
    /*
    // set input
    for(int i=0; i<N_inputs; i++){
      int il = input_layer_id[i];
//      if(data_kind==-1) layers[il].set_zero_x();
      if(data_kind==-1);
      else layers[il].set_x(((Deep_Echo_RBM)pa).data.get(data_kind, i, data_id));
    }
    */
  }
  
  void file_out_add(int data_id) throws IOException{
    for(int i=0; i<N_layers; i++){
      File file = new File("../output"+i+".txt");
      FileWriter filewriter = new FileWriter(file, true);
      String row = "";
      for(int j=0; j<layers[i].N_neurons; j++){
        row += (int)layers[i].x[j]+" ";
      }

      filewriter.write(row+"\n");
      filewriter.close();
    }

    File file = new File("../output_label.txt");
    FileWriter filewriter = new FileWriter(file, true);
    String row = "";
    float[] hoge = ((Deep_Echo_RBM)pa).data.get(0, 1, data_id);
    for(int j=0; j<10; j++){
      if(hoge[j]>0) filewriter.write(j+" ");
    }
    filewriter.write("\n");
    filewriter.close();
  }
}
