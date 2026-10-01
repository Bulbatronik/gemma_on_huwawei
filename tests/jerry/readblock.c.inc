/* Added for GemmaNano tests: readBlock(path[, position, length]) -> Uint8Array.
 * The bytes are allocated in the JerryScript heap, exactly like the watch's
 * @system.file.readArrayBuffer result, so --mem-stats shows realistic peaks. */
static jerry_value_t
gn_read_block (const jerry_value_t func_value, const jerry_value_t this_value,
               const jerry_value_t args_p[], const jerry_length_t args_cnt)
{
  (void) func_value; (void) this_value;
  if (args_cnt < 1 || !jerry_value_is_string (args_p[0]))
  {
    return jerry_create_error (JERRY_ERROR_TYPE, (const jerry_char_t *) "readBlock(path)");
  }
  char path[512];
  jerry_size_t n = jerry_string_to_utf8_char_buffer (args_p[0], (jerry_char_t *) path, sizeof (path) - 1);
  path[n] = '\0';
  FILE *f = fopen (path, "rb");
  if (f == NULL)
  {
    return jerry_create_error (JERRY_ERROR_COMMON, (const jerry_char_t *) "readBlock: cannot open");
  }
  fseek (f, 0, SEEK_END);
  long size = ftell (f);
  long pos = 0, len = size;
  if (args_cnt >= 3)
  {
    pos = (long) jerry_get_number_value (args_p[1]);
    len = (long) jerry_get_number_value (args_p[2]);
    if (pos + len > size) len = size - pos;
  }
  fseek (f, pos, SEEK_SET);
  static uint8_t buf[65536];
  if (len > (long) sizeof (buf)) len = sizeof (buf);
  size_t got = fread (buf, 1, (size_t) len, f);
  fclose (f);
  jerry_value_t ab = jerry_create_arraybuffer ((jerry_length_t) got);
  if (jerry_value_is_error (ab)) return ab;
  jerry_arraybuffer_write (ab, 0, buf, (jerry_length_t) got);
  jerry_value_t ta = jerry_create_typedarray_for_arraybuffer (JERRY_TYPEDARRAY_UINT8, ab);
  jerry_release_value (ab);
  return ta;
}

/* heapUsed() -> bytes allocated in the JS heap after a full GC (live set). */
static jerry_value_t
gn_heap_used (const jerry_value_t func_value, const jerry_value_t this_value,
              const jerry_value_t args_p[], const jerry_length_t args_cnt)
{
  (void) func_value; (void) this_value; (void) args_p; (void) args_cnt;
  jerry_heap_stats_t st;
  jerry_gc (JERRY_GC_PRESSURE_HIGH);
  if (!jerry_get_memory_stats (&st)) return jerry_create_number (-1);
  return jerry_create_number ((double) st.allocated_bytes);
}
