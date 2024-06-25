#!/bin/bash

start_time=$(date +%s)

#Archivo de entrada
INPUT_FILE="Mamografies.csv"
#Archivo de salida
OUTPUT_FILE="output.csv"

# Columnas
echo "BI-RADS,Lateralidad,Nº exploracio,Nº historial,Fecha de exploracio,Texto sin HTML" > "$OUTPUT_FILE"

# Limpiar HTLM
clean_html() {
    echo "$1" | w3m -dump -T text/html | tr -d '\n' | tr -s ' ' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/á/a/g; s/é;/e/g; s/í/i/g; s/ó/o/g; s/ú/u/g; s/ñ/n/g; s/&nbsp;/ /g; s/&amp;/&/g; s/Á/a/g; s/É;/e/g; s/Í/i/g; s/Ó/o/g; s/Ú/u/g; s/[^a-zA-Z0-9 ]/ /g' | \
    tr -s ' '
}

# Funcion para buscar BI-RADS
extract_bi_rads() {
    echo "$1" | grep -oP 'bi[- ]?rads[: ]*([0-35-6]|4\s?[abc]?)(?:2a|3b|1l|1o)?' | tr '\n' ',' | sed 's/,$//'
}

# Extraer conclusion
extract_conclusion() {
    local text="$1"
    echo "$text" | grep -oP '(conclusion|conclusiones|impresion|impresion diagnostica)[: ]+.*'
}

# Buscar lateralidad
buscar_lateralidad() {
    local text="$1"
    local lateralidad_matches
    mapfile -t lateralidad_matches < <(echo "$text" | grep -oP '\b(derecha|izquierda|esquerra|dreta|md|mi)\b')
    local -a lateralidad_norm=()

    for match in "${lateralidad_matches[@]}"; do
        if [[ "$match" =~ ^(derecha|dreta|md)$ ]]; then
            lateralidad_norm+=("mama derecha")
        elif [[ "$match" =~ ^(izquierda|esquerra|mi)$ ]]; then
            lateralidad_norm+=("mama izquierda")
        fi
    done

    if [ ${#lateralidad_norm[@]} -eq 0 ]; then
        echo "no especificada"
    else
        echo "${lateralidad_norm[@]}" | tr ' ' ',' | sed 's/,/ /g'
    fi
}

# Leer todo el archivo .csv
#counter=0
awk -F, 'NR > 1 && NR <= 6 { gsub(/"/, ""); print $1, $2, $3, $5 }' "$INPUT_FILE" | while IFS=';' read -r exploracio historia data contingut; do
    #((counter++))
    #if [ $counter -gt 10 ]; then
    #    break
    #fi

    echo "Procesado: $exploracio, $historia, $data"

    contingut_clean=$(clean_html "$contingut")
    conclusion_text=$(extract_conclusion "$contingut_clean")

    bi_rads=$(extract_bi_rads "$conclusion_text")
    [ -z "$bi_rads" ] && bi_rads="No especificado"
    lateralidad=$(buscar_lateralidad "$conclusion_text")


    # Salida
    printf "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" "$bi_rads" "$lateralidad" "$exploracio" "$historia" "$data" "${contingut_clean//\"/\"\"}" >> "$OUTPUT_FILE"
    
    #echo "$bi_rads,$lateralidad,$conclusion_text,$exploracio,$historia,$data,\"${contingut_clean//\"/\"\"}\"" >> "$OUTPUT_FILE"
    echo "Conclusiones: "
    echo "BI-RADS: $bi_rads"
    echo "Lateralidad: $lateralidad"
    echo "Nº exploracio: $exploracio"
    echo "Nº historial: $historia"
    echo "Fecha de exploracio: $data"
    echo "Texto sin HTML: $contingut_clean"
    echo "--------------"

done < <(tail -n +2 "$INPUT_FILE")

# Medir tiempo
end_time=$(date +%s)
elapse_time=$(( end_time - start_time ))

echo "Tiempo de procesamiento: $elapse_time segundos"

echo "Computational resources used:"
/usr/bin/time -v true

echo "Procesamiento completado. Archivo generado: $OUTPUT_FILE"
