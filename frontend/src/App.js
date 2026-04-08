import { useState } from "react";

// ================== CONFIGURACIÓN DE RUTAS ==================
// Detecta automáticamente si está en localhost o en el servidor de la UPM
const BASE_URL = window.location.origin; 
const API = `${BASE_URL}/ragi`;
const QUERY_PREFIX = "show me an image about ";

const suggestions = [
  "neural network architecture",
  "deep learning timeline",
  "sigmoid curve AI",
];

const categoryIcons = ["📊", "🧬", "🤖", "📈", "🔬"];

// ================== COMPONENTE: MODAL (PANTALLA COMPLETA) ==================
function ImageModal({ result, onClose }) {
  return (
    <div style={styles.modalOverlay} onClick={onClose}>
      <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <button style={styles.modalClose} onClick={onClose}>✕</button>
        <img
          /* Usamos BASE_URL porque el backend ya devuelve /ragi/images/... */
          src={`${BASE_URL}${result.image_url}`}
          alt={result.caption}
          style={styles.modalImg}
        />
        <p style={styles.modalCaption}>{result.caption}</p>
        <div style={styles.modalActions}>
          <a
            href={`${API}/download?path=${encodeURIComponent(result.image_path)}`}
            download
            style={styles.downloadBtn}
          >
            ⬇ Descargar imagen
          </a>
          <a
            href={result.source_pdf_url}
            target="_blank"
            rel="noreferrer"
            style={styles.pdfBtnModal}
          >
            📄 Ver artículo original
          </a>
        </div>
      </div>
    </div>
  );
}

// ================== COMPONENTE: SISTEMA DE ESTRELLAS ==================
function StarRating({ imagePath, query }) {
  const [selected, setSelected] = useState(0);
  const [hover, setHover] = useState(0);
  const [sent, setSent] = useState(false);

  const handleRate = async (score) => {
    setSelected(score);
    setSent(true);
    try {
      await fetch(`${API}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: imagePath,
          query,
          score,
        }),
      });
    } catch (e) {
      console.error("Error enviando valoración:", e);
    }
  };

  return (
    <div style={styles.stars}>
      {sent ? (
        <span style={styles.ratedText}>✓ Valorado ({selected}★)</span>
      ) : (
        <>
          <span style={styles.rateLabel}>¿Es relevante?</span>
          {[1, 2, 3, 4, 5].map((s) => (
            <button
              key={s}
              onClick={() => handleRate(s)}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(0)}
              style={{
                ...styles.starBtn,
                color: s <= (hover || selected) ? "#e8a020" : "#444d6e",
                transform: s <= hover ? "scale(1.2)" : "scale(1)",
              }}
            >
              ★
            </button>
          ))}
        </>
      )}
    </div>
  );
}

// ================== COMPONENTE: TARJETA DE RESULTADO ==================
function ImageCard({ result, query, onExpand }) {
  const [imgError, setImgError] = useState(false);
  const [hovered, setHovered] = useState(false);

  return (
    <div
      style={{
        ...styles.card,
        transform: hovered ? "translateY(-4px)" : "translateY(0)",
        boxShadow: hovered ? "0 12px 40px rgba(0,0,0,0.4)" : "0 2px 10px rgba(0,0,0,0.2)",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={styles.scoreTag}>
        {(result.score * 100).toFixed(0)}% relevancia
      </div>

      <div style={styles.imgWrapper} onClick={() => onExpand(result)}>
        {imgError ? (
          <div style={styles.imgPlaceholder}>Imagen no disponible</div>
        ) : (
          <>
            <img
              src={`${BASE_URL}${result.image_url}`}
              alt={result.caption}
              style={styles.cardImg}
              onError={() => setImgError(true)}
            />
            <div style={{ ...styles.imgOverlay, opacity: hovered ? 1 : 0 }}>
              <span style={styles.expandIcon}>⤢ Ver en pantalla completa</span>
            </div>
          </>
        )}
      </div>

      <div style={styles.cardBody}>
        <p style={styles.caption}>{result.caption}</p>
        <div style={styles.meta}>
          <span style={styles.metaItem}>📄 {result.doc_id}</span>
          <span style={styles.metaItem}>📑 Página {result.page}</span>
        </div>
        <div style={styles.cardLinks}>
          <a href={result.source_pdf_url} target="_blank" rel="noreferrer" style={styles.pdfLink}>
            Ver artículo →
          </a>
          <a
            href={`${API}/download?path=${encodeURIComponent(result.image_path)}`}
            download
            style={styles.downloadLink}
          >
            ⬇ Descargar
          </a>
        </div>
        <StarRating imagePath={result.image_path} query={query} />
      </div>
    </div>
  );
}

// ================== APLICACIÓN PRINCIPAL ==================
export default function App() {
  const [topic, setTopic] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);
  const [modalResult, setModalResult] = useState(null);

  const fullQuery = QUERY_PREFIX + topic;

  const handleSearch = async () => {
    if (!topic.trim()) return;
    setLoading(true);
    setError("");
    setResults([]);
    setSearched(true);

    try {
      const res = await fetch(`${API}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: fullQuery }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error en la búsqueda");
      setResults(data.results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div style={styles.root}>
      {modalResult && (
        <ImageModal result={modalResult} onClose={() => setModalResult(null)} />
      )}

      <div style={styles.gridOverlay} />

      <div style={styles.container}>
        <div style={styles.header}>
          <div style={styles.titleRow}>
            <h1 style={styles.title}>RAGI</h1>
            <span style={styles.badge}>BETA</span>
          </div>
          <p style={styles.subtitle}>
            Buscador semántico de imágenes en artículos científicos
          </p>
        </div>

        <div style={styles.searchRow}>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="neural network architectures..."
            style={styles.input}
          />
          <button onClick={handleSearch} style={styles.button} disabled={loading}>
            {loading ? "..." : "Buscar"}
          </button>
        </div>

        {!searched && (
          <>
            <div style={styles.icons}>
              {categoryIcons.map((icon, i) => (
                <span key={i} style={styles.icon}>{icon}</span>
              ))}
            </div>
            <p style={styles.description}>
              Escribe el tema de la imagen que necesitas y RAGI encontrará las más relevantes
              entre <strong style={{ color: "#8ba3c0" }}>más de 100 artículos científicos</strong> indexados.
            </p>
            <div style={styles.chips}>
              {suggestions.map((s) => (
                <button
                  key={s}
                  style={styles.chip}
                  onClick={() => setTopic(s)}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#5b7fa6")}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)")}
                >
                  {s}
                </button>
              ))}
            </div>
          </>
        )}

        {error && <div style={styles.error}>{error}</div>}

        {loading && (
          <div style={styles.loadingContainer}>
            <div style={styles.spinner} />
            <p style={styles.loadingText}>Buscando imágenes relevantes...</p>
          </div>
        )}

        {!loading && searched && results.length === 0 && !error && (
          <div style={styles.noResultsBox}>
            <span style={{ fontSize: "32px" }}>🔎</span>
            <p style={styles.noResults}>No se encontraron imágenes con suficiente relevancia.</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div style={styles.resultsSection}>
            <p style={styles.resultsHeader}>
              <span style={styles.resultsCount}>{results.length}</span>{" "}
              imagen{results.length !== 1 ? "es" : ""} encontrada
              {results.length !== 1 ? "s" : ""} para:{" "}
              <em style={{ color: "#8ba3c0" }}>"{topic}"</em>
            </p>
            <div style={styles.grid}>
              {results.map((r, i) => (
                <ImageCard key={i} result={r} query={fullQuery} onExpand={setModalResult} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ================== ESTILOS CSS-IN-JS ==================
const styles = {
  root: {
    minHeight: "100vh",
    backgroundColor: "#1a1f2e",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "center",
    paddingTop: "16vh",
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    position: "relative",
    overflowX: "hidden",
  },
  gridOverlay: {
    position: "absolute",
    inset: 0,
    backgroundImage:
      "linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
    backgroundSize: "40px 40px",
    pointerEvents: "none",
  },
  container: {
    width: "100%",
    maxWidth: 900,
    padding: "0 24px 80px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 28,
    zIndex: 1,
  },
  header: { textAlign: "center" },
  titleRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    marginBottom: 10,
  },
  title: {
    fontSize: 56,
    fontWeight: 900,
    letterSpacing: "0.12em",
    color: "#ffffff",
    margin: 0,
    textTransform: "uppercase",
  },
  badge: {
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.1em",
    color: "#8ba3c0",
    border: "1px solid #3a4f66",
    borderRadius: 4,
    padding: "3px 8px",
    alignSelf: "flex-start",
    marginTop: 8,
  },
  subtitle: {
    color: "#8ba3c0",
    fontSize: 15,
    margin: 0,
    letterSpacing: "0.01em",
  },
  searchRow: {
    display: "flex",
    width: "100%",
    gap: 12,
  },
  input: {
    flex: 1,
    backgroundColor: "#212840",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 10,
    padding: "14px 20px",
    fontSize: 15,
    color: "#e0e8f0",
    outline: "none",
    caretColor: "#4e7fba",
  },
  button: {
    backgroundColor: "#3b6ea5",
    color: "#ffffff",
    border: "none",
    borderRadius: 10,
    padding: "14px 28px",
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  icons: {
    display: "flex",
    gap: 20,
    fontSize: 26,
    marginTop: 60,
  },
  icon: { opacity: 0.75 },
  description: {
    color: "#5e7a96",
    fontSize: 14,
    textAlign: "center",
    lineHeight: 1.7,
    margin: 0,
  },
  chips: {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
    justifyContent: "center",
  },
  chip: {
    backgroundColor: "transparent",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 999,
    color: "#8ba3c0",
    padding: "7px 18px",
    fontSize: 13,
    cursor: "pointer",
    transition: "border-color 0.2s",
  },
  error: {
    backgroundColor: "#2a1a1a",
    border: "1px solid #5a2020",
    color: "#ff6b6b",
    padding: "12px 16px",
    borderRadius: "8px",
    fontSize: "14px",
    width: "100%",
  },
  loadingContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "60px 0",
    gap: "16px",
  },
  spinner: {
    width: "40px",
    height: "40px",
    border: "3px solid #2a3556",
    borderTop: "3px solid #3b6ea5",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  loadingText: { color: "#5e7a96", fontSize: "14px" },
  noResultsBox: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "60px 0",
    gap: "12px",
  },
  noResults: { color: "#5e7a96", fontSize: "15px", margin: 0 },
  resultsSection: { width: "100%" },
  resultsHeader: { fontSize: "14px", color: "#5e7a96", marginBottom: "16px" },
  resultsCount: { color: "#8ba3c0", fontWeight: "700" },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
    gap: "24px",
  },
  card: {
    backgroundColor: "#212840",
    borderRadius: "16px",
    border: "1px solid rgba(255,255,255,0.08)",
    overflow: "hidden",
    position: "relative",
    transition: "transform 0.2s, box-shadow 0.2s",
  },
  scoreTag: {
    position: "absolute",
    top: "12px",
    right: "12px",
    backgroundColor: "#3b6ea5",
    color: "#fff",
    fontSize: "11px",
    fontWeight: "700",
    padding: "4px 10px",
    borderRadius: "20px",
    zIndex: 1,
  },
  imgWrapper: {
    position: "relative",
    cursor: "pointer",
    overflow: "hidden",
  },
  cardImg: {
    width: "100%",
    height: "220px",
    objectFit: "contain",
    backgroundColor: "#1a1f2e",
    padding: "12px",
    display: "block",
  },
  imgOverlay: {
    position: "absolute",
    inset: 0,
    backgroundColor: "rgba(0,0,0,0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "opacity 0.2s",
  },
  expandIcon: {
    color: "#fff",
    fontSize: "14px",
    fontWeight: "600",
    backgroundColor: "rgba(59,110,165,0.9)",
    padding: "8px 16px",
    borderRadius: "8px",
  },
  imgPlaceholder: {
    width: "100%",
    height: "220px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#1a1f2e",
    color: "#555",
    fontSize: "13px",
  },
  cardBody: { padding: "16px" },
  caption: {
    fontSize: "13px",
    color: "#7a8fa8",
    lineHeight: "1.6",
    marginBottom: "12px",
  },
  meta: { display: "flex", gap: "12px", marginBottom: "10px" },
  metaItem: { fontSize: "12px", color: "#3a4f66" },
  cardLinks: {
    display: "flex",
    gap: "16px",
    alignItems: "center",
    marginBottom: "14px",
  },
  pdfLink: { fontSize: "13px", color: "#5b8fc4", textDecoration: "none" },
  downloadLink: { fontSize: "13px", color: "#5e7a96", textDecoration: "none" },
  stars: {
    display: "flex",
    alignItems: "center",
    gap: "2px",
    borderTop: "1px solid rgba(255,255,255,0.06)",
    paddingTop: "12px",
  },
  rateLabel: { fontSize: "12px", color: "#3a4f66", marginRight: "6px" },
  starBtn: {
    background: "none",
    border: "none",
    fontSize: "20px",
    cursor: "pointer",
    padding: "0 2px",
    transition: "color 0.1s, transform 0.1s",
  },
  ratedText: { fontSize: "13px", color: "#e8a020" },
  modalOverlay: {
    position: "fixed",
    inset: 0,
    backgroundColor: "rgba(0,0,0,0.85)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    padding: "24px",
  },
  modalContent: {
    backgroundColor: "#212840",
    borderRadius: "16px",
    border: "1px solid rgba(255,255,255,0.08)",
    padding: "24px",
    maxWidth: "800px",
    width: "100%",
    position: "relative",
    maxHeight: "90vh",
    overflowY: "auto",
  },
  modalClose: {
    position: "absolute",
    top: "16px",
    right: "16px",
    background: "none",
    border: "none",
    color: "#5e7a96",
    fontSize: "20px",
    cursor: "pointer",
  },
  modalImg: {
    width: "100%",
    maxHeight: "500px",
    objectFit: "contain",
    backgroundColor: "#1a1f2e",
    borderRadius: "8px",
    padding: "12px",
    marginBottom: "16px",
  },
  modalCaption: {
    fontSize: "14px",
    color: "#7a8fa8",
    lineHeight: "1.6",
    marginBottom: "16px",
  },
  modalActions: { display: "flex", gap: "16px" },
  downloadBtn: {
    padding: "10px 20px",
    backgroundColor: "#3b6ea5",
    color: "#fff",
    borderRadius: "8px",
    textDecoration: "none",
    fontSize: "14px",
    fontWeight: "600",
  },
  pdfBtnModal: {
    padding: "10px 20px",
    backgroundColor: "#2a3556",
    color: "#e0e8f0",
    borderRadius: "8px",
    textDecoration: "none",
    fontSize: "14px",
  },
};