    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.7) -> bool:
        """
        Perform fuzzy matching between query and target strings.
        
        Args:
            query: User query string
            target: Target metadata string to match against
            threshold: Similarity threshold (0.0-1.0) where 1.0 is exact match
            
        Returns:
            Boolean indicating whether the strings match according to the threshold
        """
        # Simple word overlap score
        query_words = set(query.lower().split())
        target_words = set(target.lower().split())
        
        # Skip very short targets
        if len(target_words) < 2:
            return False
            
        # Calculate word overlap
        common_words = query_words.intersection(target_words)
        if not common_words:
            return False
            
        # Calculate similarity score
        similarity = len(common_words) / max(len(query_words), len(target_words))
        
        # Look for consecutive word matches which are stronger indicators
        query_bigrams = self._get_bigrams(query.lower())
        target_bigrams = self._get_bigrams(target.lower())
        common_bigrams = set(query_bigrams).intersection(set(target_bigrams))
        
        # Boost score if we have consecutive word matches
        if common_bigrams:
            similarity += 0.2 * (len(common_bigrams) / max(len(query_bigrams), len(target_bigrams)))
            
        return similarity >= threshold
        
    def _get_bigrams(self, text: str) -> List[str]:
        """Get bigrams (consecutive word pairs) from text"""
        words = text.split()
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        
    def _search_lineage_db(self, query: str) -> Dict[str, Any]:
        """
        Search for relevant metadata in the lineage database
        
        Args:
            query: User query string
            
        Returns:
            Dict with search results
        """
        # Use the search_lineage_database tool
        try:
            return search_lineage_database(query)
        except Exception as e:
            logger.error(f"Error searching lineage database: {e}")
            return {"success": False, "error": str(e)}
