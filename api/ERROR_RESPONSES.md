# Ответы об ошибках API

При ошибках API возвращается **HTTP 200** и в теле — JSON с полем **error**. Так клиенту не нужно обрабатывать 4xx/5xx отдельно: достаточно прочитать тело и проверить наличие `error`.

## Формат при ошибке (HTTP 200)

```json
{
  "error": "Ошибка валидации запроса",
  "detail": "current_message: Field required; llm_model_name: Field required",
  "code": "validation_error",
  "errors": [
    { "field": "current_message", "message": "Field required", "type": "missing" },
    { "field": "llm_model_name", "message": "Field required", "type": "missing" }
  ]
}
```

- **error** — краткое описание.
- **detail** — подробное описание (также в заголовке `X-Error-Detail`).
- **code** — код ошибки (машиночитаемый).
- **errors** — при валидации: список по полям (`field`, `message`, `type`).

## Обработка на клиенте

Всегда получаете 200 — проверяйте тело:

```java
// Java: читайте поток как обычно (getInputStream()), затем проверьте JSON
String body = readResponse(conn.getInputStream());
JSONObject json = new JSONObject(body);
if (json.has("error")) {
    String msg = json.optString("detail", json.getString("error"));
    if (json.has("errors")) { /* разбор по полям */ }
    throw new RuntimeException(msg);
}
// иначе используйте json.get("content"), json.get("model") и т.д.
```

```javascript
// JavaScript / fetch
const res = await fetch(url, { method: 'POST', body: JSON.stringify(data) });
const json = await res.json();
if (json.error) {
  console.error(json.error, json.detail, json.errors);
} else {
  console.log(json.content);
}
```
