set xrange [0:$0]
set title "Loyalty Fluctuation"
set xlabel "Day"
set ylabel "Number loyal"
set key below
plot "logs/loyalty.dat" using 1:2 with lines title "Manufacturer 0",\
"logs/loyalty.dat" using 1:3 with lines title "Manufacturer 1"
pause 1
reread
