
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
   to_head('..'), 
   to_cor(),
   to_begin(),

   #Conv 1
   to_Conv( name='conv1',x_filter = 8, z_filter = '', offset="(0,0,0)", to="(0,0,0)", width=2, height=32, depth=32),

   to_BN(name="bn1", x_filter = 8, z_filter = 256, to="(conv1-east)", offset="(1.2,0,0)",  width=0.4, height=32, depth=32),
   to_connection("conv1", "bn1"),                # arrow pool → BN

   to_ReLU(name="relu1", x_filter = 8, z_filter = 256, width=0.4, height=32, depth=32, to="(bn1-east)", offset="(1.5,0,0)"),
   to_connection("bn1", "relu1"),                  # arrow BN → ReLU

   to_Conv( name='conv2',x_filter = 8, z_filter = 256, offset="(1.5,0,0)", to="(relu1-east)", width=2, height=32, depth=32),
   to_connection("relu1", "conv2"),

   to_BN(name="bn2", x_filter = 8, z_filter = 256, to="(conv2-east)", offset="(1.2,0,0)",  width=0.4, height=32, depth=32),
   to_connection("conv2", "bn2"),                # arrow pool → BN

   to_Conv( name='conv3',x_filter = 8, z_filter = 256, offset="(2,0,0)", to="(bn2-east)", width=2, height=32, depth=32),
   to_connection("bn2", "conv3"),

   # Sum sphere
   to_Sum(name="sum1", to="(conv3-east)", offset="(1.4,0,0)"),
   to_connection("conv3", "sum1"),
   to_skip(of="conv1", to="sum1", xoffset="10"),

   to_end() 
   ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
