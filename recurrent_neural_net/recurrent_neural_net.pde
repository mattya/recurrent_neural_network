import java.awt.Frame;

PFrame pf;
secondApplet sap;

void setup() {
  size(Wid*30, H*8);
   PFrame pf = new PFrame();
  background(255);
  imgs = new PImage[N_images];
  for(int i=0; i<N_images; i++){
    imgs[i] = loadImage("../fonts/"+i+".png");
  }
  alloc();
  img_to_d();
  random_init();
}

int step = 0;
int disp_im = 0;
void draw() {
  println("step: "+step);
  train();
  
  float[][][][] res = new float[N_layer][NNMax][Wid][H];
  float error = 0;
  for(int im=0; im<min(10, N_images); im++){
    calc_all_recurrent(im, res);
    disp_func(im, res);
//    error += calc_error(im, res);
  }
  println(error);
  stroke(0);
  for(int i=0; i<30; i++){
    line(Wid*i, 0, Wid*i, height);
    line(0, H*i, width, H*i);
  }
  
  calc_all_recurrent(disp_im, res);
  draw_tree(disp_im, res);
  sap.redraw();
  step++;
//  if(step>=1024) noLoop();
  
  sap.stroke(255, 0, 0);
  sap.line(0, 42*cur_last_layer, 400, 42*cur_last_layer);
}

void keyPressed(){
  if(key=='a'){
    disp_im++;
    if(disp_im>=N_images) disp_im = 0;
  }else if(key=='n'){
    cur_last_layer++;
  }
}

public class PFrame extends Frame {
    public PFrame() {
        setBounds(100,100,1280,840);
        sap = new secondApplet();
        add(sap);
        sap.init();
        show();
    }
}

public class secondApplet extends PApplet {
    public void setup() {
        size(1280, 840);
        noLoop();
    }

    public void draw() {
    }
}
