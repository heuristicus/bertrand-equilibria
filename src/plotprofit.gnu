set xrange [0:$0]
set title "Daily Profits"
set xlabel "Day"
set ylabel "Profit (pence)"
set key below
plot "logs/profit.dat" using 1:2 with lines title "Manufacturer 0",\
"logs/profit.dat" using 1:3 with lines title "Manufacturer 1"
pause 1
reread
