python make_1d_cbc_bank.py --psd-model aLIGOZeroDetHighPower \
       --min-frequency 30 \
       --min-total-mass 2 \
       --max-total-mass 300 \
       --sample-rate 2048 \
       --min-match 0.97 \
       --waveform-model SEOBNRv4_ROM \
       --output-bank template_bank_2_300_30Hz.h5 \
       --output-plot /home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/TemplateBank/template_bank_2_300_30Hz.png \
       --spacing-tolerance 0.1
