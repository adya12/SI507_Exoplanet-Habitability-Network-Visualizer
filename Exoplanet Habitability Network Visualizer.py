import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pyvis.network import Network
import requests
from io import StringIO
import time
import json

# Configuration with optimized physics for gentle movement
CONFIG = {
    'max_planets': 150,
    'min_similarity': 0.85,
    'earth_color': '#00FF00',
    'node_base_size': 20,
    'earth_size': 50,
    'edge_width_multiplier': 1.2,
    'physics': {
        'enabled': True,
        'stabilization': {
            'enabled': True,
            'iterations': 200
        },
        'barnesHut': {
            'gravitationalConstant': -2500,
            'centralGravity': 0.7,
            'springLength': 200,
            'damping': 0.6,
            'avoidOverlap': 0.3
        },
        'minVelocity': 0.15
    }
}

class ExoplanetNetwork:
    def __init__(self):
        self.df = None
        self.graph = nx.Graph()
        self.analysis_results = {}

    def fetch_data(self):
        """Fetch exoplanet data from NASA with Earth forced in"""
        print("🛰️ Fetching exoplanet data...")
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + \
              "query=select+pl_name,st_teff,pl_rade,pl_orbper,pl_eqt,pl_bmasse+" + \
              "from+pscomppars+where+pl_rade+>+0.5+and+pl_bmasse+>+0.1&format=csv"
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            self.df = pd.read_csv(StringIO(response.text))
            print(f"✅ Loaded {len(self.df)} exoplanets")
        except Exception as e:
            print(f"⚠️ Error: {e}\n🔄 Using local data")
            self.df = pd.read_csv("exoplanet_data.csv")

        # Force Earth into dataset with standard values
        earth_data = {
            'pl_name': 'Earth',
            'st_teff': 288,
            'pl_rade': 1.0,
            'pl_orbper': 365.25,
            'pl_eqt': 288,
            'pl_bmasse': 1.0
        }
        self.df = pd.concat([self.df, pd.DataFrame([earth_data])], ignore_index=True)
        
        if len(self.df) > CONFIG['max_planets']:
            # Ensure Earth is always included in the sample
            earth_row = self.df[self.df['pl_name'] == 'Earth']
            other_rows = self.df[self.df['pl_name'] != 'Earth'].sample(
                CONFIG['max_planets']-1, 
                random_state=42
            )
            self.df = pd.concat([earth_row, other_rows])
            print(f"📊 Using {CONFIG['max_planets']} planet sample (including Earth)")

    def preprocess_data(self):
        """Clean and normalize data with Earth protection"""
        print("\n🧹 Preprocessing...")
        features = ['pl_name', 'st_teff', 'pl_rade', 'pl_orbper', 'pl_eqt', 'pl_bmasse']
        self.df = self.df[features].dropna()
        
        # Double-check Earth exists
        if 'Earth' not in self.df['pl_name'].values:
            print("⚠️ Earth missing - re-adding")
            earth_data = {
                'pl_name': 'Earth',
                'st_teff': 288,
                'pl_rade': 1.0,
                'pl_orbper': 365.25,
                'pl_eqt': 288,
                'pl_bmasse': 1.0
            }
            self.df = pd.concat([self.df, pd.DataFrame([earth_data])], ignore_index=True)
        
        scaler = StandardScaler()
        self.df[features[1:]] = scaler.fit_transform(self.df[features[1:]])
        print(f"📈 Prepared {len(self.df)} planets (Earth included)")

    def build_similarity_network(self):
        """Create network with guaranteed Earth connections"""
        print("\n🔗 Building network...")
        n_neighbors = min(15, len(self.df)-1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nbrs.fit(self.df.iloc[:, 1:])
        distances, indices = nbrs.kneighbors(self.df.iloc[:, 1:])
        similarities = 1 - distances

        # Create edges with Earth getting extra connections
        for i in range(len(indices)):
            planet_name = self.df.iloc[i]['pl_name']
            for j, idx in enumerate(indices[i]):
                if i != idx and (similarities[i, j] > CONFIG['min_similarity'] or planet_name == 'Earth'):
                    self.graph.add_edge(
                        planet_name,
                        self.df.iloc[idx]['pl_name'],
                        weight=similarities[i, j],
                        title=f"Similarity: {similarities[i,j]:.2f}"
                    )
        
        # Force Earth connections if missing
        if 'Earth' not in self.graph.nodes():
            print("⚠️ Adding Earth with artificial connections")
            earth_features = self.df[self.df['pl_name'] == 'Earth'].iloc[0, 1:].values.reshape(1, -1)
            distances, indices = nbrs.kneighbors(earth_features)
            for j, idx in enumerate(indices[0]):
                self.graph.add_edge(
                    'Earth',
                    self.df.iloc[idx]['pl_name'],
                    weight=1 - distances[0][j],
                    title=f"Similarity: {1-distances[0][j]:.2f}"
                )

        print(f"🌐 Network: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")

    def visualize(self):
        """Create interactive visualization with controlled movement"""
        print("\n🎨 Generating visualization...")
        nt = Network(
            height="800px",
            width="100%",
            bgcolor="#0f172a",  # Dark background
            font_color="white",
            notebook=True,
            cdn_resources='remote',
            directed=False
        )

        # Add Earth with special styling
        if 'Earth' in self.graph.nodes():
            nt.add_node(
                'Earth',
                color=CONFIG['earth_color'],
                size=CONFIG['earth_size'],
                title="⭐ <b>EARTH</b> ⭐<br>Reference Planet<br>Radius: 1.0 Earth<br>Temp: 288K",
                shape="star",
                borderWidth=5,
                x=0,  # Initial position at center
                y=0,
                fixed=False,  # Allows slight movement
                physics=True,
                mass=10  # Heavier node stays more central
            )

        # Add other planets
        for node in self.graph.nodes():
            if node == 'Earth':
                continue
                
            earth_sim = self._get_earth_similarity(node)
            nt.add_node(
                node,
                color=self._get_color(earth_sim),
                size=self._get_size(earth_sim),
                title=self._create_tooltip(node, earth_sim),
                shape="dot",
                physics=True
            )

        # Add edges with similarity-based styling
        for u, v, data in self.graph.edges(data=True):
            nt.add_edge(
                u, v,
                width=data['weight'] * CONFIG['edge_width_multiplier'],
                color=f"rgba(150, 150, 150, {data['weight']*0.7})",
                title=data.get('title', '')
            )

        # Configure physics for natural but controlled movement
        options = {
            "physics": {
                "enabled": True,
                "stabilization": CONFIG['physics']['stabilization'],
                "barnesHut": CONFIG['physics']['barnesHut'],
                "minVelocity": CONFIG['physics']['minVelocity']
            },
            "nodes": {
                "font": {
                    "size": 14,
                    "face": "Arial",
                    "strokeWidth": 2
                },
                "scaling": {
                    "min": 10,
                    "max": 30
                }
            },
            "edges": {
                "smooth": {
                    "type": "continuous",
                    "roundness": 0.2
                },
                "hoverWidth": 1.5
            },
            "interaction": {
                "hover": True,
                "tooltipDelay": 200
            }
        }
        nt.set_options(json.dumps(options))
        
        # Add explanatory legend
        self._add_legend(nt)

        output_file = "exoplanet_network.html"
        nt.show(output_file)
        print(f"\n✅ Visualization saved to {output_file}")
        print("🌍 Earth is the large green star at center")
        print("🔄 Network has gentle physics - drag nodes to explore")

    def _get_earth_similarity(self, node):
        """Calculate similarity to Earth (0-1 scale)"""
        if node == 'Earth':
            return 1.0
        if 'Earth' in self.graph.neighbors(node):
            return self.graph.edges[node, 'Earth']['weight']
        return 0.0

    def _get_color(self, similarity):
        """Color gradient from red to yellow"""
        r = int(255 * (1 - similarity))
        g = int(255 * similarity)
        return f"rgb({r}, {g}, 0)"

    def _get_size(self, similarity):
        """Size based on Earth similarity"""
        return CONFIG['node_base_size'] + (15 * similarity)

    def _create_tooltip(self, node, similarity):
        """Generate hover tooltip with planet info"""
        features = self.df[self.df['pl_name'] == node].iloc[0]
        return (
            f"<b>{node}</b><br>"
            f"Similarity to Earth: {similarity:.0%}<br>"
            f"Radius: {features['pl_rade']:.1f}× Earth<br>"
            f"Temperature: {features['pl_eqt']:.0f} K<br>"
            f"Orbital Period: {features['pl_orbper']:.1f} days"
        )

    def _add_legend(self, net):
        """Add explanatory legend to visualization"""
        legend_nodes = [
            ("legend", "EXOPLANET HABITABILITY NETWORK", 25, "#ffffff"),
            ("earth_exp", "⭐ = EARTH (Reference Planet)", 20, CONFIG['earth_color']),
            ("similar_exp", "🔴 → 🟡 = Similarity to Earth", 20, "#ffcc00"),
            ("interaction", "Drag nodes to explore relationships", 16, "#aaaaaa")
        ]
        
        for i, (node_id, label, font_size, color) in enumerate(legend_nodes):
            net.add_node(
                node_id,
                label=label,
                color="rgba(0,0,0,0)",
                font={"size": font_size, "color": color},
                x=-1000,
                y=-400 + i*50,
                fixed=True,
                physics=False
            )

def main():
    print("\n" + "="*50)
    print(" EXOPLANET HABITABILITY NETWORK VISUALIZER ")
    print("="*50)
    
    start_time = time.time()
    analyzer = ExoplanetNetwork()
    
    analyzer.fetch_data()
    analyzer.preprocess_data()
    analyzer.build_similarity_network()
    analyzer.visualize()
    
    print(f"\nTotal runtime: {time.time()-start_time:.1f} seconds")

if __name__ == "__main__":
    main()