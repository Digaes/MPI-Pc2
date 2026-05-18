from mpi4py import MPI
import os
import time
from collections import Counter


def cargar_consulta(consulta_path, case_sensitive=False):
    """
    Lee consulta.txt y devuelve un conjunto de palabras objetivo.
    """
    if not os.path.isfile(consulta_path):
        raise FileNotFoundError(f"No se encontró el archivo de consulta: {consulta_path}")

    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]

    if not case_sensitive:
        palabras = [w.lower() for w in palabras]

    return set(palabras)

def contar_palabras_en_corpus(dataset_dir, consulta_name="consulta.txt", case_sensitive=False):
    """
    Cuenta cuántas veces aparecen las palabras de consulta.txt
    en todos los archivos file_*.txt del directorio dataset_dir.
    """
    consulta_path = os.path.join(dataset_dir, consulta_name)
    palabras_objetivo = cargar_consulta(consulta_path, case_sensitive=case_sensitive)

    freq_global = Counter()
    archivos_procesados = 0
    total_tokens_leidos = 0

    for fname in os.listdir(dataset_dir):
        if not fname.startswith("file_") or not fname.endswith(".txt"):
            continue

        ruta = os.path.join(dataset_dir, fname)
        if not os.path.isfile(ruta):
            continue

        archivos_procesados += 1

        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                palabras = linea.split()

                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]

                total_tokens_leidos += len(palabras)

                for w in palabras:
                    if w in palabras_objetivo:
                        freq_global[w] += 1

    return freq_global, archivos_procesados, total_tokens_leidos


def contar_palabras_en_archivos(lista_archivos, palabras_objetivo, case_sensitive=False):
    freq_global = Counter()
    archivos_procesados = 0
    total_tokens_leidos = 0

    for ruta in lista_archivos:
        if not os.path.isfile(ruta):
            continue

        archivos_procesados += 1

        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                palabras = linea.split()

                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]

                total_tokens_leidos += len(palabras)

                for w in palabras:
                    if w in palabras_objetivo:
                        freq_global[w] += 1

    return freq_global, archivos_procesados, total_tokens_leidos


def guardar_resultados_csv(resultados_path, freq_global):
    """
    Guarda el conteo global en CSV para comparar con MPI y Dask.
    """
    with open(resultados_path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for palabra in sorted(freq_global):
            f.write(f"{palabra},{freq_global[palabra]}\n")

# ---------------------------------------------------------------------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "dataset")

consulta_name = "consulta.txt"
case_sensitive = False
top_n = 10
output_file = "baseline_results.csv"
start_total = time.perf_counter()

#(1)rank 0 reads consulta.txt
if rank == 0:
    consulta_path = os.path.join(dataset_dir,consulta_name)
    try:
        palabras_objetivo = cargar_consulta(consulta_path,case_sensitive=case_sensitive)
        query_list=list(palabras_objetivo)
    except FileNotFoundError as e:
        print("Error:", e)
        query_list = None
    
    #(3)rank 0 obtains the list of file_*.txt files
    all_files=[
        os.path.join(dataset_dir,fname)
        for fname in os.listdir(dataset_dir)
        if fname.startswith("file_") and fname.endswith(".txt") and os.path.isfile(os.path.join(dataset_dir,fname))
    ]
else: 
    query_list = None
    all_files = None

# (2)rank 0 broadcasts the query words to all processes using broadcast
query_list = comm.bcast(query_list,root=0)
if query_list is None:
    exit(1)
palabras_objetivo = set(query_list)

# (4)the files are distributed statically among the processes, 
all_files = comm.bcast(all_files, root=0)
my_files= all_files[rank::size]
print(f"Rank {rank}: assigned {len(my_files)} files")

# (5) each process counts locally the occurrences of the query words in its assigned files
local_start = time.perf_counter()
freq_local, archivos_local,tokens_local = contar_palabras_en_archivos(my_files,palabras_objetivo,case_sensitive = case_sensitive)
local_time= time.perf_counter() - local_start
print(f"rank {rank}: local processing time = {local_time:.6f} seconds")

# (6) partial results are gathered in rank 0
all_counts = comm.gather(dict(freq_local), root=0)
all_archivos = comm.gather(archivos_local, root=0)
all_tokens = comm.gather(tokens_local, root=0)

# (7) rank 0 builds the global result and prints the top 10
if rank == 0: 
    freq_global = Counter()
    for partial in all_counts:
        freq_global.update(partial)

    archivos_procesados = sum(all_archivos)
    total_tokens_leidos = sum(all_tokens)
    total_occurrences = sum(freq_global.values())
    top_words = freq_global.most_common(top_n)


    total_time = time.perf_counter() - start_total

    resultados_path = os.path.join(dataset_dir, output_file)
    guardar_resultados_csv(resultados_path, freq_global)

    print(f"Tiempo de ejecución: {total_time:.6f} segundos\n")
    print(f"Dataset procesado: {dataset_dir}")
    print(f"Archivo de consulta: {consulta_name}")
    print(f"Archivos procesados: {archivos_procesados}")
    print(f"Total de tokens leídos: {total_tokens_leidos}")
    print(f"Total de ocurrencias encontradas: {total_occurrences}")
    print(f"Resultados guardados en: {resultados_path}\n")

    print(f"Top {top_n} palabras de consulta en el corpus:")
    for palabra, cuenta in top_words:
        print(f"  {palabra}: {cuenta}")

    print(f"EXECUTION_TIME={total_time:.6f}")