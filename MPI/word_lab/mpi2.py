from mpi4py import MPI 
import os 
import time 
from collections import Counter

def cargar_consulta(consulta_path, case_sensitive=False):
    if not os.path.isfile(consulta_path):
        raise FileNotFoundError(f"No se encontró el archivo de consulta: {consulta_path}")
    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]
    if not case_sensitive:
        palabras = [w.lower() for w in palabras]
    return set(palabras)

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
    with open(resultados_path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for palabra in sorted(freq_global):
            f.write(f"{palabra},{freq_global[palabra]}\n")

# ---------------------------------------------------------------------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_WORK   = 1 
TAG_RESULT = 2
TAG_STOP   = 3

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(SCRIPT_DIR, "dataset")
CONSULTA_NAME  = "consulta.txt"
CASE_SENSITIVE = False
TOP_N          = 10
CHUNK_SIZE     = 10

start_total = time.perf_counter()

if rank == 0:
    consulta_path = os.path.join(DATASET_DIR, CONSULTA_NAME)
    palabras_objetivo = cargar_consulta(consulta_path, case_sensitive=CASE_SENSITIVE)
    query_list = list(palabras_objetivo)
    all_files = [
        os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
        if f.startswith("file_") and f.endswith(".txt")
        and os.path.isfile(os.path.join(DATASET_DIR, f))
    ]
    if size < 2:
        freq_global, archivos_procesados, total_tokens_leidos = contar_palabras_en_archivos(
            all_files, palabras_objetivo, case_sensitive=CASE_SENSITIVE
        )
        total_ocurrencias = sum(freq_global.values())
        top_words = freq_global.most_common(TOP_N)
        resultados_path = os.path.join(DATASET_DIR, "mpi2_results.csv")
        guardar_resultados_csv(resultados_path, freq_global)

        total_time = time.perf_counter() - start_total
        print(f"\nTiempo de ejecución: {total_time:.6f} segundos")
        print(f"Dataset procesado: {DATASET_DIR}")
        print(f"Archivo de consulta: {CONSULTA_NAME}")
        print(f"Archivos procesados: {archivos_procesados}")
        print(f"Total de tokens leídos: {total_tokens_leidos}")
        print(f"Total de ocurrencias encontradas: {total_ocurrencias}")
        print(f"Resultados guardados en: {resultados_path}\n")
        print(f"Top {TOP_N} palabras de consulta en el corpus:")
        for palabra, cuenta in top_words:
            print(f"  {palabra}: {cuenta}")
        print(f"EXECUTION_TIME={total_time:.6f}")
        exit(0)

    print(f"Master: {len(all_files)} files | {len(query_list)} query words | {size-1} workers")
else:
    query_list = None
    all_files  = None

query_list = comm.bcast(query_list, root=0)
palabras_objetivo = set(query_list)

# Master (rank 0) 

if rank == 0:
    file_index     = 0
    active_workers = size - 1
    freq_global    = Counter()
    archivos_procesados  = 0
    total_tokens_leidos  = 0
    files_per_worker = {i: 0 for i in range(1, size)}

    for worker in range(1, size):
        chunk = all_files[file_index : file_index + CHUNK_SIZE]
        if chunk:
            comm.send(chunk, dest=worker, tag=TAG_WORK)
            files_per_worker[worker] += len(chunk)
            file_index += len(chunk)
        else:
            comm.send(None, dest=worker, tag=TAG_STOP)
            active_workers -= 1

    while active_workers > 0:
        status = MPI.Status()
        partial_counts, partial_tokens, n_files = comm.recv(
            source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status
        )
        src = status.Get_source()
        freq_global.update(partial_counts)
        archivos_procesados += n_files
        total_tokens_leidos += partial_tokens

        chunk = all_files[file_index : file_index + CHUNK_SIZE]
        if chunk:
            comm.send(chunk, dest=src, tag=TAG_WORK)
            files_per_worker[src] += len(chunk)
            file_index += len(chunk)
        else:
            comm.send(None, dest=src, tag=TAG_STOP)
            active_workers -= 1

    print("\nFiles processed per worker:")
    for w in sorted(files_per_worker):
        print(f"  Worker {w}: {files_per_worker[w]} files")

    total_ocurrencias = sum(freq_global.values())
    top_words         = freq_global.most_common(TOP_N)
    resultados_path   = os.path.join(DATASET_DIR, "mpi2_results.csv")
    guardar_resultados_csv(resultados_path, freq_global)
    total_time = time.perf_counter() - start_total

    print(f"\nTiempo de ejecución: {total_time:.6f} segundos")
    print(f"Archivos procesados: {archivos_procesados}")
    print(f"Total de tokens leídos: {total_tokens_leidos}")
    print(f"Total de ocurrencias encontradas: {total_ocurrencias}")
    print(f"Resultados guardados en: {resultados_path}\n")
    print(f"Top {TOP_N} palabras de consulta en el corpus:")
    for palabra, cuenta in top_words:
        print(f"  {palabra}: {cuenta}")
    print(f"EXECUTION_TIME={total_time:.6f}")

# WORKERS (rank > 0)

else:
    files_processed = 0
    local_work_time = 0.0

    while True:
        status  = MPI.Status()
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            break

        t0 = time.perf_counter()
        freq_local, n_arch, tokens = contar_palabras_en_archivos(
            payload, palabras_objetivo, case_sensitive=CASE_SENSITIVE
        )
        local_work_time += time.perf_counter() - t0
        files_processed += n_arch

        comm.send((dict(freq_local), tokens, n_arch), dest=0, tag=TAG_RESULT)

    print(f"Rank {rank}: {files_processed} files | local work time = {local_work_time:.6f}s")