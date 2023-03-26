
import io
import numpy as np
import sqlite3

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

conn = sqlite3.connect('./data/fma_preprocessed/preprocessed.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = conn.cursor()
cursor.execute("SELECT idx, path FROM tokens WHERE idx=?", (0,))
print(cursor.fetchone())

for i in [3,-1]:
    cursor.execute("SELECT clap, semantic, coarse FROM tokens WHERE idx=?", (i,))

    data = cursor.fetchone()

    if data is None:
        print("No data found")
    else:
        for datum in data:
            print(datum.shape)

        print(data[1])
