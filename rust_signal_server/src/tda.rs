//! TDA Wasserstein distance tracker.
//!
//! Computes H0 persistence diagram of a delay-embedding of log-returns,
//! then tracks the Wasserstein-1 distance between consecutive diagrams.
//! High distance = topology of return distribution is actively shifting
//! = regime transition in progress.
//!
//! Implementation: simplified Vietoris-Rips H0 via union-find on sorted edges.
//! Full ripser port is Phase 2 — this version is correct and fast enough for PoC.

pub struct WassersteinTracker {
    window:   usize,
    prev_dgm: Vec<f32>,   // previous H0 persistence values
}

impl WassersteinTracker {
    pub fn new(window: usize) -> Self {
        Self { window, prev_dgm: Vec::new() }
    }

    /// Update with latest log-return slice, return Wasserstein-1 distance.
    pub fn update(&mut self, log_ret: &[f32]) -> f32 {
        let n = log_ret.len();
        if n < self.window { return 0.0; }
        let slice = &log_ret[n - self.window..];

        // Z-score normalise so point-cloud distances are scale-invariant.
        // Raw M1 log returns are ~1e-4; without this step tanh rounds to 0.
        let mean = slice.iter().sum::<f32>() / slice.len() as f32;
        let std  = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / slice.len() as f32).sqrt().max(1e-8);
        let normed: Vec<f32> = slice.iter().map(|x| (x - mean) / std).collect();

        let dgm  = h0_persistence(&normed, 3);
        let dist = if self.prev_dgm.is_empty() {
            0.0
        } else {
            wasserstein1(&dgm, &self.prev_dgm)
        };
        self.prev_dgm = dgm;
        // With unit-std inputs typical distances are O(1-5); scale 0.1 spreads [0,1]
        (dist * 0.1).tanh()
    }
}

/// H0 persistence via Vietoris-Rips on delay-embedded point cloud.
/// dim: delay embedding dimension (3 gives good results at window=60).
fn h0_persistence(x: &[f32], dim: usize) -> Vec<f32> {
    let n = x.len();
    if n < dim { return vec![]; }
    let n_pts = n - dim + 1;

    // Build point cloud: each point is (x[i], x[i+1], ..., x[i+dim-1])
    let pts: Vec<Vec<f32>> = (0..n_pts)
        .map(|i| x[i..i+dim].to_vec())
        .collect();

    // Pairwise distances
    let mut edges: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..n_pts {
        for j in i+1..n_pts {
            let d = euclidean(&pts[i], &pts[j]);
            edges.push((d, i, j));
        }
    }
    edges.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());

    // Union-find for H0
    let mut parent: Vec<usize> = (0..n_pts).collect();
    let birth:  Vec<f32>   = vec![0.0; n_pts];
    let mut persist: Vec<f32>  = Vec::new();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x { parent[x] = find(parent, parent[x]); }
        parent[x]
    }

    for (d, i, j) in &edges {
        let ri = find(&mut parent, *i);
        let rj = find(&mut parent, *j);
        if ri != rj {
            // Merge: younger component dies
            let bi = birth[ri];
            let bj = birth[rj];
            if bi <= bj {
                parent[rj] = ri;
                persist.push(d - bj);
            } else {
                parent[ri] = rj;
                persist.push(d - bi);
            }
        }
    }
    persist
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x,y)|(x-y).powi(2)).sum::<f32>().sqrt()
}

/// Wasserstein-1 distance between two persistence diagrams (sorted 1D matching).
fn wasserstein1(a: &[f32], b: &[f32]) -> f32 {
    let mut aa = a.to_vec(); aa.sort_by(|x,y| x.partial_cmp(y).unwrap());
    let mut bb = b.to_vec(); bb.sort_by(|x,y| x.partial_cmp(y).unwrap());
    // Pad shorter with zeros
    while aa.len() < bb.len() { aa.push(0.0); }
    while bb.len() < aa.len() { bb.push(0.0); }
    aa.iter().zip(&bb).map(|(a,b)|(a-b).abs()).sum::<f32>()
        / aa.len().max(1) as f32
}
