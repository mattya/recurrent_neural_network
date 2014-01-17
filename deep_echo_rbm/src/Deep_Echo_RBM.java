import java.io.IOException;

import processing.core.*;

public class Deep_Echo_RBM extends PApplet{
  HW_digits data;
  Network nn;
  
  public void setup(){
    data = new HW_digits(this);
    nn = new Network(loadStrings(Global.architecture_file_name), this);
    
    size(Global.width, Global.height);
    background(128);
    frameRate(2000);
  }
  
  int offset_x = 50;
  int offset_y = 10;
  int cur_data_num = 0;
  int cur_state = 0;
  int cur_data_t = 0;
  int cur_repeat = 0;
  int cur_step = 0;
  public void draw(){
    
    if(cur_step%73==0){
      nn.one_step(cur_state>=1?0:-1, cur_data_num+cur_data_t, cur_state>=1?0:1, cur_state);
      
      println(cur_step, cur_repeat, cur_state, cur_data_t);
      for(int i=0; i<nn.N_weights; i++){
        print(" "+nn.weights[i].infinity_norm()+" ");
      }
      println();
      offset_y = 10;
      for(int i=0; i<nn.N_layers; i++){
        fill(0);
        text(nn.layers[i].name, 10, offset_y+10);
        PImage img_l = nn.layers[i].visualize();
        image(img_l, offset_x, offset_y);
        offset_y += img_l.height+10;
        
 //       println(i, img_l.width, img_l.height, offset_x, offset_y);
      }
      
      offset_x += Global.wid+1;
      if(offset_x > Global.width-10) offset_x = 50;
  
      try {
        nn.file_out_add(cur_data_num+cur_data_t);
      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }

      cur_data_t++;
      if(cur_data_t==28){
        if(cur_repeat==1){
          cur_data_num = 28*(int)random(0, data.N_data/28);
          cur_repeat = 0;
          cur_data_t = 0;
          cur_state = (cur_state+1)%6;
          cur_step++;
        }else{
          cur_data_t = 0;
          cur_repeat++;
        }
      }
    }else{
      while(cur_step%73!=0){
        nn.one_step(cur_state>=1?0:-1, cur_data_num+cur_data_t, cur_state>=1?0:1, cur_state);
        cur_data_t++;
        if(cur_data_t==28){
          if(cur_repeat==1){
            cur_data_num = 28*(int)random(0, data.N_data/28);
            cur_repeat = 0;
            cur_data_t = 0;
            cur_state = (cur_state+1)%6;
            cur_step++;
          }else{
            cur_data_t = 0;
            cur_repeat++;
          }
        }
      }
    }
    
    
  }
}
