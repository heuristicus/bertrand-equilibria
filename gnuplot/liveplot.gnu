set xrange [0:30]
set yrange [0:900]
plot "plot.dat" using 1:2 with lines
replot;
pause 1;
reread
