using Revise
using DataFrames
using Dates

import XLSX
xfile = XLSX.readxlsx("YM_de_opt.xlsx")
data_sheet = xfile["hist"] 
df = DataFrame(XLSX.gettable(data_sheet))

df[!, :dates] = Date.(df.dates)

plot(df, x=:dates, y=:ipi)

X = convert(Array{Float64}, Array(df[1:end-2,2:end]))