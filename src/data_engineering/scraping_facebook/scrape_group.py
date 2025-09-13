"""
facebook_scraper_raw_extractor.py
Versión: V9.0 - Producción y Persistencia con SQLite
Objetivo: Modificar el scraper v8.3 para que escriba los resultados
          directamente a una base de datos SQLite en lugar de un CSV.
          Esto permite ejecuciones incrementales (ej. diarias) para construir
          un corpus masivo a lo largo del tiempo sin duplicar trabajo.
          
          Utiliza el 'text_hash' como PRIMARY KEY para una deduplicación
          automática y eficiente.

Requisitos: playwright, pandas
"""

import os
import re
import time
import json
import logging
import hashlib
import random
import sqlite3 # <-- CAMBIO: Importamos SQLite
from collections import defaultdict

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ------------------ CONFIG ------------------
GROUP_URL = "https://www.facebook.com/share/g/17AeWmU4yr/"
COOKIES_PATH = "facebook_cookies.json"
# OUTPUT_CSV = "facebook_corpus_raw..." <-- YA NO SE USA CSV
DATABASE_NAME = "cucei_reviews.db" # <-- CAMBIO: Output a Base de Datos
POST_SELECTOR = "[role='article']"
SCROLL_COUNT = 50 
HEADLESS = True
MIN_TEXT_LENGTH = 15

NOISE_PREFIX_REGEX = r'^(Miembro anónimo\s*\d*\s*|Colaborador en ascenso\s*|Respuesta destacada\s*|Conoce del tema\s*|Autor original\s*|Admin\s*|Moderador\s*|Miembro activo\s*|Ver m\u00e1s\s*|Recomendado\s*)'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------------ UTILS ------------------

def text_hash(s: str) -> str:
    return hashlib.sha1(s.strip().encode('utf-8')).hexdigest()

# ------------------ DATABASE (NUEVO) ------------------

def setup_database():
    """Crea la conexión a la BD y la tabla 'reviews' si no existe."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        # Creamos la tabla. text_hash es la clave única para evitar duplicados.
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            text_hash TEXT PRIMARY KEY,
            post_index_visible INTEGER,
            post_url TEXT,
            origin_type TEXT,
            raw_text TEXT,
            scraped_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        logging.info(f"Base de datos '{DATABASE_NAME}' conectada y tabla 'reviews' asegurada.")
        return conn
    except Exception as e:
        logging.error(f"Error fatal conectando o creando la base de datos: {e}")
        if conn:
            conn.close()
        return None

def save_record_to_db(conn, record, stats):
    """
    Intenta insertar un registro en la BD.
    Usa 'INSERT OR IGNORE' para que si el text_hash (PRIMARY KEY) ya existe,
    la transacción simplemente se ignore sin causar un error.
    """
    try:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO reviews (text_hash, post_index_visible, post_url, origin_type, raw_text)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            record['text_hash'], 
            record['post_index_visible'], 
            record['post_url'], 
            record['origin_type'], 
            record['raw_text']
        ))
        conn.commit()
        
        # cursor.rowcount nos dirá si una fila fue REALMENTE insertada (1) o ignorada (0)
        if cursor.rowcount > 0:
            stats['nuevas_unidades_guardadas_db'] += 1
        else:
            stats['duplicados_ignorados_por_db'] += 1
            
    except Exception as e:
        logging.error(f"Error guardando registro en DB: {e}")
        stats['errores_db'] += 1

# ------------------ EXPANSION (Lógica V8.3 sin cambios) ------------------

def click_all_within(locator, selector_texts, max_clicks=40):
    clicks = 0
    for sel in selector_texts:
        try:
            elements = locator.locator(sel)
            count = elements.count()
            for i in range(count):
                try:
                    elements.nth(i).click(timeout=2500)
                    clicks += 1
                    time.sleep(0.25)
                    if clicks >= max_clicks:
                        return clicks
                except Exception:
                    continue
        except Exception:
            continue
    return clicks

def expand_post_detail(page, post_element):
    clicks = 0
    post_url = None
    try:
        all_links = post_element.locator("a[href*='/posts/'], a[href*='/permalink.php'], a[role='link']").all()
        for link in all_links:
            href = link.get_attribute('href')
            if href and ('/posts/' in href or '/permalink.php' in href or '/groups/' in href):
                if 'permalink' in href or '/posts/' in href:
                    post_url = href
                    break
                post_url = href 
        if not post_url:
             post_url = post_element.locator("a[role='link']").first.get_attribute('href')
    except Exception:
        post_url = None

    if post_url:
        try:
            detail = page.context.new_page()
            detail.goto(post_url, wait_until='domcontentloaded', timeout=15000) 
            for _ in range(30): 
                c1 = click_all_within(detail, ["text=/Ver m\u00e1s comentarios/i", "text=/See more comments/i", "text=/Mostrar m\u00e1s comentarios/i"], max_clicks=8)
                c2 = click_all_within(detail, ["text=/Ver respuestas/i", "text=/See replies/i", "text=/Mostrar respuestas/i"], max_clicks=30)
                c3 = click_all_within(detail, ["text=/Ver m\u00e1s/i", "text=/See more/i", "text=/Mostrar m\u00e1s/i"], max_clicks=40)
                clicks += (c1 + c2 + c3)
                time.sleep(0.5)
                if c1 == 0 and c2 == 0 and c3 == 0:
                    break
            
            content = detail.locator('body').inner_text(timeout=15000) # (Corrección V8.3)
            detail.close()
            return content, clicks, post_url
        
        except Exception as e:
            logging.warning(f"Error abriendo detalle {post_url}. Fallback. Error: {e}")
            try:
                click_all_within(post_element, ["text=/Ver m\u00e1s comentarios/i","text=/Ver respuestas/i","text=/Ver m\u00e1s/i"], max_clicks=30)
                return post_element.inner_text(timeout=10000), clicks, post_url
            except Exception as e_fallback:
                 logging.error(f"Fallback falló para {post_url}. Error: {e_fallback}")
                 return "", clicks, post_url
    else:
        try:
            click_all_within(post_element, ["text=/Ver m\u00e1s comentarios/i","text=/Ver respuestas/i","text=/Ver m\u00e1s/i"], max_clicks=30)
            return post_element.inner_text(timeout=10000), clicks, "no_url_found"
        except Exception as e_local:
            logging.error(f"Expansión local falló. Error: {e_local}")
            return "", clicks, "no_url_found"

def extract_units_from_detail_text(detail_text: str):
    if not detail_text:
        return []
    detail_text = re.sub(NOISE_PREFIX_REGEX, '', detail_text, flags=re.IGNORECASE)
    detail_text = re.sub(r'\s+(Me gusta|Comentar|Compartir|Responder)\s*', '\n', detail_text, flags=re.IGNORECASE)
    detail_text = re.sub(r'\n{2,}', '\n', detail_text).strip()

    units = []
    lines = [l.strip() for l in detail_text.split('\n') if l.strip()]
    if not lines:
        return units

    units.append({'origin': 'post', 'text': lines[0]})
    buffer = []
    for ln in lines[1:]:
        if len(ln) < 3:
            continue
        if re.match(r'^[A-Z\s\u00c0-\u017f\.\-]{3,}$', ln) and len(ln.split()) <= 4:
            buffer = [ln]
            continue
        if buffer:
            name_candidate = ' '.join(buffer)
            combined = f"{name_candidate}: {ln}"
            units.append({'origin': 'comment', 'text': combined})
            buffer = []
            continue
        m = re.match(r'^([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})\s*[:\-]\s*(.+)$', ln)
        if m:
            units.append({'origin': 'comment', 'text': f"{m.group(1)}: {m.group(2)}"})
            continue
        units.append({'origin': 'comment', 'text': ln})
    return units

# ------------------ SCRAPING (V9.0 - Conectado a DB) ------------------

def scrape_group_raw():
    # results = [] <-- CAMBIO: Ya no acumulamos en memoria
    # seen_hashes = set() <-- CAMBIO: La BD maneja los duplicados
    stats = defaultdict(int)

    # --- CAMBIO: SETUP DB ---
    conn = setup_database()
    if conn is None:
        logging.error("No se pudo inicializar la base de datos. Abortando.")
        return

    with sync_playwright() as p:
        logging.info(f'Lanzando Chromium (Headless: {HEADLESS})...')
        browser = p.chromium.launch(headless=HEADLESS, slow_mo=0)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36")

        # Carga de Cookies (sin cambios)
        try:
            with open(COOKIES_PATH, 'r', encoding='utf-8') as f: cookies = json.load(f)
            for c in cookies:
                if 'sameSite' in c and c['sameSite'] not in ['Strict','Lax','None']: c.pop('sameSite', None)
            context.add_cookies(cookies)
            logging.info('Cookies cargadas.')
        except Exception:
            logging.warning('Cookies no encontradas o inválidas.')

        page = context.new_page()
        try:
            page.goto(GROUP_URL, wait_until='domcontentloaded', timeout=60000)
            page.wait_for_selector(POST_SELECTOR, timeout=20000)
            logging.info('Página de grupo cargada.')
        except Exception as e:
            logging.error(f'Error crítico abriendo grupo: {e}')
            browser.close(); conn.close()
            return

        # Scroll Robusto (sin cambios)
        logging.info(f'Iniciando scroll robusto (max {SCROLL_COUNT} intentos)...')
        last_height = page.evaluate('document.body.scrollHeight')
        stuck_count = 0
        for i in range(SCROLL_COUNT):
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            try: page.wait_for_load_state('networkidle', timeout=3000)
            except Exception: time.sleep(1.0) 
            new_height = page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                stuck_count += 1
                logging.warning(f'Scroll {i+1}/{SCROLL_COUNT}: Altura no cambió. "Atasco" {stuck_count}/5')
            else:
                stuck_count = 0; last_height = new_height
                logging.info(f'Scroll {i+1}/{SCROLL_COUNT}: Nueva altura detectada.')
            if stuck_count >= 5:
                logging.warning('Scroll detenido: Página dejó de cargar contenido.'); break

        # Iteración Just-in-Time (sin cambios)
        post_count = page.locator(POST_SELECTOR).count()
        logging.info(f'Encontrados {post_count} posts visibles. Iniciando procesamiento profundo...')

        for idx in range(post_count):
            stats['total_posts_intentados'] += 1
            try:
                post_element = page.locator(POST_SELECTOR).nth(idx)

                if not post_element.is_visible(timeout=5000):
                    logging.warning(f"Post #{idx} no visible (DOM reciclado). Saltando.")
                    stats['posts_saltados_no_visibles'] += 1
                    continue
                
                detail_text, clicks, post_url = expand_post_detail(page, post_element)
                stats['expand_clicks_totales'] += clicks
                
                units = extract_units_from_detail_text(detail_text)
                if not units:
                    stats['posts_vacios_o_error_extraccion'] += 1; continue

                for unit in units:
                    text = unit['text'].strip()
                    text = re.sub(NOISE_PREFIX_REGEX, '', text, flags=re.IGNORECASE).strip()

                    if len(text) < MIN_TEXT_LENGTH:
                        stats['unidades_cortas_descartadas'] += 1; continue

                    h = text_hash(text)
                    # if h in seen_hashes: <-- CAMBIO: La BD lo hace por nosotros.
                    
                    record = {
                        'text_hash': h, # <-- La BD necesita el hash
                        'post_index_visible': idx,
                        'post_url': post_url,
                        'origin_type': unit['origin'],
                        'raw_text': text
                    }
                    
                    # --- CAMBIO: GUARDAR DIRECTO A DB ---
                    save_record_to_db(conn, record, stats)
                    # results.append(record) <-- Ya no se usa
                
                # Pausa Humana (sin cambios)
                time.sleep(random.uniform(2.0, 5.0)) 

            except Exception as e:
                logging.error(f'Error procesando post #{idx}: {e}')
                stats['errores_post_loop'] += 1
                time.sleep(random.uniform(5.0, 10.0))
            
            if (idx+1) % 10 == 0:
                 logging.info(f'Procesados {idx+1}/{post_count} posts...')

        browser.close()

    # --- CAMBIO: CERRAR DB Y FINALIZAR ---
    if conn:
        conn.close()

    logging.info('Scraping finalizado. Conexión a DB cerrada.')

    # Ya no guardamos CSV. Mostramos el resumen de la base de datos.
    logging.info('----- Resumen de Ejecución (V9.0 - SQLite) -----')
    for k, v in stats.items():
        logging.info(f'{k}: {v}')
    
    logging.info("Para exportar los datos a CSV (para tu notebook), usa un visor de SQLite o Pandas:")
    logging.info("df = pd.read_sql_query('SELECT * FROM reviews', sqlite3.connect('cucei_reviews.db'))")


if __name__ == '__main__':
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass 

    logging.info('INICIANDO SCRAPER (V9.0) - Persistencia con SQLite')
    scrape_group_raw()