1. If you have a .pvd time series

python plot_max_disp.py series.pvd --var displacement


2. If you just have a directory of frames

python plot_max_disp.py ./frames --var displacement --assume-dt 0.01


3. Save to file

python plot_max_disp.py series.pvd --out max_disp.png
