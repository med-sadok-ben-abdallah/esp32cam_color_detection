/*
  ESP32-CAM — Détection de couleur/objet simple, entièrement sur l'appareil
  (aucun traitement dans le cloud : c'est un exemple pratique de Edge AI / TinyML "léger")
  AI-Thinker ESP32-CAM

  Principe :
  - La caméra capture des images au format RGB565 (permet de lire directement
    les composantes Rouge/Vert/Bleu de chaque pixel, sans décodage JPEG coûteux).
  - Une tâche FreeRTOS parcourt les pixels (avec échantillonnage pour rester
    rapide) et cherche ceux qui ressemblent à une couleur cible.
  - Elle calcule le "centroïde" (position moyenne) des pixels correspondants :
    c'est la position approximative de l'objet recherché dans l'image.
  - Le résultat (position + nombre de pixels détectés) est protégé par un Mutex
    et exposé via une page web (polling JSON) et optionnellement via MQTT.

  Bibliothèques nécessaires : esp32-camera (incluse), PubSubClient (MQTT, optionnel)
  Board : AI Thinker ESP32-CAM
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "esp_http_server.h"

// ---------- Configuration réseau ----------
const char* WIFI_SSID     = "TON_WIFI";
const char* WIFI_PASSWORD = "TON_MOT_DE_PASSE";
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "esp32cam/color";
const bool  MQTT_ENABLED  = true; // passe à false si tu ne veux pas utiliser MQTT

// ---------- Pinout AI-Thinker ESP32-CAM ----------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ---------- Paramètres de détection (modifiables à chaud via /setcolor) ----------
volatile uint8_t targetR = 220;   // couleur cible par défaut : rouge
volatile uint8_t targetG = 40;
volatile uint8_t targetB = 40;
volatile uint8_t colorTolerance = 60;  // distance de couleur acceptée (0-255)
const int SAMPLE_STEP = 2;             // on analyse 1 pixel sur SAMPLE_STEP dans chaque dimension
const int MIN_MATCHING_PIXELS = 25;    // seuil minimal pour dire "objet détecté"

// ---------- Résultat de détection, protégé par Mutex ----------
struct DetectionResult {
  bool detected;
  float x;        // position horizontale normalisée (0.0 = gauche, 1.0 = droite)
  float y;        // position verticale normalisée (0.0 = haut, 1.0 = bas)
  int matchCount; // nombre de pixels correspondant à la couleur cible
};
DetectionResult latestDetection = { false, 0, 0, 0 };
SemaphoreHandle_t detectionMutex;

httpd_handle_t webServer = NULL;
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ---------- Initialisation caméra ----------
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;   config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href  = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; // accès direct aux couleurs, pas de décodage JPEG
  config.frame_size   = FRAMESIZE_QQVGA;  // 160x120 : nécessaire pour rester rapide en RGB565
  config.fb_count     = 2;

  return esp_camera_init(&config) == ESP_OK;
}

// ---------- Distance approximative entre deux couleurs ----------
int colorDistance(int r1, int g1, int b1, int r2, int g2, int b2) {
  return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2); // distance "Manhattan", suffisante et rapide
}

// =====================================================
// Task capture + analyse (Core 1)
// =====================================================
void taskColorDetection(void* pvParameters) {
  for (;;) {
    camera_fb_t* fb = esp_camera_fb_get();

    if (fb != NULL && fb->format == PIXFORMAT_RGB565) {
      long sumX = 0, sumY = 0;
      int matchCount = 0;
      int width = fb->width;
      int height = fb->height;

      for (int y = 0; y < height; y += SAMPLE_STEP) {
        for (int x = 0; x < width; x += SAMPLE_STEP) {
          int pixelIndex = (y * width + x) * 2; // 2 octets par pixel en RGB565
          uint16_t pixel = (fb->buf[pixelIndex] << 8) | fb->buf[pixelIndex + 1];
          // NOTE MATÉRIELLE : l'ordre des octets peut varier selon le module caméra.
          // Si les couleurs détectées semblent inversées (bleu au lieu de rouge...),
          // inverser l'ordre ci-dessus : (fb->buf[pixelIndex+1] << 8) | fb->buf[pixelIndex]

          uint8_t r5 = (pixel >> 11) & 0x1F;
          uint8_t g6 = (pixel >> 5)  & 0x3F;
          uint8_t b5 =  pixel        & 0x1F;
          uint8_t r8 = (r5 << 3) | (r5 >> 2); // remise à l'échelle 5 bits -> 8 bits
          uint8_t g8 = (g6 << 2) | (g6 >> 4); // remise à l'échelle 6 bits -> 8 bits
          uint8_t b8 = (b5 << 3) | (b5 >> 2);

          if (colorDistance(r8, g8, b8, targetR, targetG, targetB) < colorTolerance) {
            sumX += x;
            sumY += y;
            matchCount++;
          }
        }
      }

      xSemaphoreTake(detectionMutex, portMAX_DELAY);
      latestDetection.matchCount = matchCount;
      if (matchCount >= MIN_MATCHING_PIXELS) {
        latestDetection.detected = true;
        latestDetection.x = (float)sumX / matchCount / width;
        latestDetection.y = (float)sumY / matchCount / height;
      } else {
        latestDetection.detected = false;
      }
      xSemaphoreGive(detectionMutex);

      esp_camera_fb_return(fb);
    } else if (fb != NULL) {
      esp_camera_fb_return(fb);
    }

    vTaskDelay(pdMS_TO_TICKS(120)); // ~8 analyses/seconde, largement suffisant pour ce cas d'usage
  }
}

// =====================================================
// Task réseau (Core 0) : publie le résultat sur MQTT
// =====================================================
void taskMqttPublish(void* pvParameters) {
  if (!MQTT_ENABLED) {
    vTaskDelete(NULL);
  }

  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);

  for (;;) {
    if (!mqttClient.connected()) {
      mqttClient.connect("esp32cam-color");
    }
    mqttClient.loop();

    DetectionResult snapshot;
    xSemaphoreTake(detectionMutex, portMAX_DELAY);
    snapshot = latestDetection;
    xSemaphoreGive(detectionMutex);

    char payload[96];
    snprintf(payload, sizeof(payload),
             "{\"detected\":%s,\"x\":%.2f,\"y\":%.2f,\"pixels\":%d}",
             snapshot.detected ? "true" : "false", snapshot.x, snapshot.y, snapshot.matchCount);
    mqttClient.publish(MQTT_TOPIC, payload);

    vTaskDelay(pdMS_TO_TICKS(500)); // publication 2x/seconde : suffisant, évite de saturer le broker
  }
}

// =====================================================
// Handler HTTP : "/" -> page web avec un point qui suit l'objet détecté
// =====================================================
esp_err_t indexHandler(httpd_req_t* req) {
  const char* html =
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-CAM Couleur</title></head>"
    "<body style='background:#111;color:#eee;font-family:sans-serif;text-align:center;'>"
    "<h3>Détection de couleur en direct</h3>"
    "<div style='position:relative;width:320px;height:240px;margin:auto;background:#222;border:1px solid #444;'>"
    "<div id='dot' style='position:absolute;width:16px;height:16px;border-radius:50%;"
    "background:red;display:none;transform:translate(-50%,-50%);'></div>"
    "</div>"
    "<p id='status'>En attente...</p>"
    "<script>"
    "setInterval(async () => {"
    "  const r = await fetch('/detection'); const d = await r.json();"
    "  const dot = document.getElementById('dot');"
    "  const status = document.getElementById('status');"
    "  if (d.detected) {"
    "    dot.style.display = 'block';"
    "    dot.style.left = (d.x * 320) + 'px';"
    "    dot.style.top  = (d.y * 240) + 'px';"
    "    status.textContent = 'Objet détecté (' + d.pixels + ' pixels)';"
    "  } else {"
    "    dot.style.display = 'none';"
    "    status.textContent = 'Aucun objet de cette couleur';"
    "  }"
    "}, 200);"
    "</script></body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, strlen(html));
}

// =====================================================
// Handler HTTP : "/detection" -> résultat JSON (interrogé par la page web)
// =====================================================
esp_err_t detectionHandler(httpd_req_t* req) {
  DetectionResult snapshot;
  xSemaphoreTake(detectionMutex, portMAX_DELAY);
  snapshot = latestDetection;
  xSemaphoreGive(detectionMutex);

  char json[96];
  snprintf(json, sizeof(json),
           "{\"detected\":%s,\"x\":%.2f,\"y\":%.2f,\"pixels\":%d}",
           snapshot.detected ? "true" : "false", snapshot.x, snapshot.y, snapshot.matchCount);
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, json, strlen(json));
}

// =====================================================
// Handler HTTP : "/setcolor?r=NN&g=NN&b=NN&tol=NN" -> change la couleur cible à chaud
// =====================================================
esp_err_t setColorHandler(httpd_req_t* req) {
  char query[64], valueStr[8];

  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "r", valueStr, sizeof(valueStr)) == ESP_OK) targetR = atoi(valueStr);
    if (httpd_query_key_value(query, "g", valueStr, sizeof(valueStr)) == ESP_OK) targetG = atoi(valueStr);
    if (httpd_query_key_value(query, "b", valueStr, sizeof(valueStr)) == ESP_OK) targetB = atoi(valueStr);
    if (httpd_query_key_value(query, "tol", valueStr, sizeof(valueStr)) == ESP_OK) colorTolerance = atoi(valueStr);
    Serial.printf("Nouvelle couleur cible : R=%d G=%d B=%d (tolérance=%d)\n", targetR, targetG, targetB, colorTolerance);
  }

  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_send(req, "OK", 2);
}

void startWebServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 4;

  if (httpd_start(&webServer, &config) == ESP_OK) {
    httpd_uri_t indexUri     = { "/",           HTTP_GET, indexHandler,     NULL };
    httpd_uri_t detectionUri = { "/detection",  HTTP_GET, detectionHandler, NULL };
    httpd_uri_t setColorUri  = { "/setcolor",   HTTP_GET, setColorHandler,  NULL };
    httpd_register_uri_handler(webServer, &indexUri);
    httpd_register_uri_handler(webServer, &detectionUri);
    httpd_register_uri_handler(webServer, &setColorUri);
  }
}

// =====================================================
// Setup / Loop
// =====================================================
void setup() {
  Serial.begin(115200);

  if (!initCamera()) {
    Serial.println("Erreur d'initialisation de la caméra");
    return;
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Page de détection : http://");
  Serial.println(WiFi.localIP());

  detectionMutex = xSemaphoreCreateMutex();

  xTaskCreatePinnedToCore(taskColorDetection, "TaskColor", 4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(taskMqttPublish,     "TaskMqtt",  4096, NULL, 1, NULL, 0);

  startWebServer();
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000)); // tout le travail se fait dans les tâches et le serveur HTTP
}
