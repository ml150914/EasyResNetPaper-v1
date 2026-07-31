
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( '../../Figures/TTmap.png' ),

    #Conv 1
    to_Conv( name='conv1',x_filter = 8, z_filter = 256, offset="(0,0,0)", to="(0,0,0)", width=2, height=32, depth=32),

    #to_Pool(name="pool_b1", offset="(0,0,0)", to="(conv1-east)", width=1, height=28, depth=28, opacity=0.5),
    #to_connection("conv1", "pool_b1"),              # arrow conv → pool

    to_BN(name="bn1", x_filter = 8, z_filter = 256, to="(conv1-east)", offset="(1.2,0,0)",  width=0.4, height=32, depth=32),
    to_connection("conv1", "bn1"),                # arrow pool → BN

    to_ReLU(name="relu1", x_filter = 8, z_filter = 256, width=0.4, height=32, depth=32, to="(bn1-east)", offset="(1.5,0,0)"),
    to_connection("bn1", "relu1"),                  # arrow BN → ReLU

    to_ResBlock(
    name="res1",
    x_filter=8,
    z_filter=128,
    to="(relu1-east)",
    offset="(2,0,0)",
    width=4,
    height=32,
    depth=32,
    caption="ResBlock1"
    ),
    to_connection("relu1", "res1"),                 # arrow into Residual bloc

    to_ResBlock(
    name="res2",
    x_filter=16,
    z_filter=64,
    to="(res1-east)",
    offset="(2,0,0)",
    width=8,
    height=16,
    depth=16,
    caption="ResBlock2"
    ),

    to_connection(of="res1", to="res2"),

    to_ResBlock(
    name="res3",
    x_filter=32,
    z_filter=32,
    to="(res2-east)",
    offset="(1.5,0,0)",
    width=16,
    height=8,
    depth=8,
    caption="ResBlock3"
    ),

    #to_skip(of="res1", to="res2", pos=1.25),
    to_connection(of="res2", to="res3"),


    to_GAP(name="gap", x_filter=1, z_filter=32,
       to="(res3-east)", offset="(1,0,0)",
       width=2, height=2, depth=8),

    to_connection("res3", "gap"),

    to_Dropout(name="drop", x_filter=1, z_filter=32,
       to="(gap-east)", offset="(1,0,0)",
       width=2, height=2, depth=8),

    to_connection("gap", "drop"),

    to_FullyConnected(name="fullyconn", input_size=32, output_size=2,
       to="(drop-east)", offset="(1,0,0)",
       width=2, height=2, depth=8),

    to_connection("drop", "fullyconn"),

    #Batch Norm

    #ReLu

    #ResBlock1

    #ResBlock2

    #Global Average Pooling

    #Drop Out

    #Fully Connnected

    #block-001
    #to_Conv( name='ccr_b1', s_filer=256, n_filer=128, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40  ),
    #to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),
    
    #*block_2ConvPool( name='b2', botton='pool_b1', top='pool_b2', s_filer=256, n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ),
    #*block_2ConvPool( name='b3', botton='pool_b2', top='pool_b3', s_filer=128, n_filer=256, offset="(1,0,0)", size=(25,25,4.5), opacity=0.5 ),
    #*block_2ConvPool( name='b4', botton='pool_b3', top='pool_b4', s_filer=64,  n_filer=512, offset="(1,0,0)", size=(16,16,5.5), opacity=0.5 ),

    #Bottleneck
    #block-005
    #to_ConvConvRelu( name='ccr_b5', s_filer=32, n_filer=(1024,1024), offset="(2,0,0)", to="(pool_b4-east)", width=(8,8), height=8, depth=8, caption="Bottleneck"  ),
    #to_connection( "pool_b4", "ccr_b5"),

    #Decoder
    #*block_Unconv( name="b6", botton="ccr_b5", top='end_b6', s_filer=64,  n_filer=512, offset="(2.1,0,0)", size=(16,16,5.0), opacity=0.5 ),
    #to_skip( of='ccr_b4', to='ccr_res_b6', pos=1.25),
    #*block_Unconv( name="b7", botton="end_b6", top='end_b7', s_filer=128, n_filer=256, offset="(2.1,0,0)", size=(25,25,4.5), opacity=0.5 ),
    #to_skip( of='ccr_b3', to='ccr_res_b7', pos=1.25),    
    #*block_Unconv( name="b8", botton="end_b7", top='end_b8', s_filer=256, n_filer=128, offset="(2.1,0,0)", size=(32,32,3.5), opacity=0.5 ),
    #to_skip( of='ccr_b2', to='ccr_res_b8', pos=1.25),    
    
    #*block_Unconv( name="b9", botton="end_b8", top='end_b9', s_filer=512, n_filer=64,  offset="(2.1,0,0)", size=(40,40,2.5), opacity=0.5 ),
    #to_skip( of='ccr_b1', to='ccr_res_b9', pos=1.25),
    
    #to_ConvSoftMax( name="soft1", s_filer=512, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40, caption="SOFT" ),
    #to_connection( "end_b9", "soft1"),

    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
