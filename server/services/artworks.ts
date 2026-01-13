import axios from "axios";

export interface ScrapedArtwork {
  title: string;
  year: string;
  imageUrl: string;
  source: string;
  metadata: any;
}

export async function fetchArtworks(artistName: string): Promise<ScrapedArtwork[]> {
  const results: ScrapedArtwork[] = [];
  
  try {
    // 1. Met Museum
    const metUrl = `https://collectionapi.metmuseum.org/public/collection/v1/search?q=${encodeURIComponent(artistName)}&hasImages=true`;
    const metSearch = await axios.get(metUrl);
    
    if (metSearch.data.objectIDs) {
      const ids = metSearch.data.objectIDs.slice(0, 15); // Take first 15 to be fast
      for (const id of ids) {
        try {
          const detail = await axios.get(`https://collectionapi.metmuseum.org/public/collection/v1/objects/${id}`);
          if (detail.data.primaryImageSmall) {
            results.push({
              title: detail.data.title,
              year: detail.data.objectDate || "Unknown",
              imageUrl: detail.data.primaryImageSmall,
              source: "Metropolitan Museum of Art",
              metadata: { objectID: id }
            });
          }
        } catch (e) { continue; }
      }
    }
  } catch (e) {
    console.error("Met API Error", e);
  }

  // If we need more, try Art Institute of Chicago
  if (results.length < 30) {
    try {
      const aicUrl = `https://api.artic.edu/api/v1/artworks/search?q=${encodeURIComponent(artistName)}&limit=20&fields=id,title,date_display,image_id`;
      const aicSearch = await axios.get(aicUrl);
      
      for (const item of aicSearch.data.data) {
        if (item.image_id) {
          results.push({
            title: item.title,
            year: item.date_display || "Unknown",
            imageUrl: `https://www.artic.edu/iiif/2/${item.image_id}/full/843,/0/default.jpg`,
            source: "Art Institute of Chicago",
            metadata: { id: item.id }
          });
        }
      }
    } catch (e) {
      console.error("AIC API Error", e);
    }
  }

  // If we still need more, try Cleveland
  if (results.length < 30) {
    try {
      const clevelandUrl = `https://openaccess-api.clevelandart.org/api/artworks/?q=${encodeURIComponent(artistName)}&limit=20&has_image=1`;
      const clevelandSearch = await axios.get(clevelandUrl);
      
      if (clevelandSearch.data.data) {
         for (const item of clevelandSearch.data.data) {
           if (item.images && item.images.web && item.images.web.url) {
             results.push({
               title: item.title,
               year: item.creation_date || "Unknown",
               imageUrl: item.images.web.url,
               source: "Cleveland Museum of Art",
               metadata: { id: item.id }
             });
           }
         }
      }
    } catch (e) {
      console.error("Cleveland API Error", e);
    }
  }

  return results.slice(0, 30);
}
