import processing.core.*;

//
//  x[i][j][k]: #k of keiretsu j of i_th data
//
public class Data {
  PApplet pa;
  int N_data;
  int N_var;
  
  int[] N_var_dim;
  float[][][] x;
  
  Data(){};
  
  void initialize(int Nd0, int Nv0, int[] Nvd0){
    N_data = Nd0;
    N_var =Nv0;
    N_var_dim = new int[N_var];
    for(int i=0; i<N_var; i++) N_var_dim[i] = Nvd0[i];

    x = new float[N_data][N_var][1];
    for(int j=0; j<N_data; j++){
      for(int i=0; i<N_var; i++){
        x[j][i] =new float[N_var_dim[i]];
      }
    }
  }
}

class HW_digits extends Data{
  int[] Nvd0 = {28, 10};

  HW_digits(PApplet pa0) {
    pa = pa0;
    initialize(40000*28, 2, Nvd0);
    String[] file = pa.loadStrings("/home/mattya/Dropbox/kaneko_ken/2013_winter/train_data/handwritten_digits_normal.txt");
    for(int i=0; i<N_data/28; i++){
      String[] spl = pa.split(file[i], ',');
      for(int t=0; t<28; t++){
        for(int j=0; j<10; j++){
          
          if(Integer.parseInt(spl[0])==j) x[i*28+t][1][j]=1;  //1
          else x[i*28+t][1][j]=-1;  //-1
        }
        for(int j=0; j<28; j++){
          x[i*28+t][0][j] = pa.map(Integer.parseInt(spl[1+t*28+j]), 0, 256, -1, 1);
        }
      }
    }
    // TODO Auto-generated constructor stub
  }

  public float[] get(int data_kind, int i, int data_id) {
    return x[data_id][i];
  }

}
