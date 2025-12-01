import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.
        Use the structured metadata provided for each retrieved product.

        ## Instructions:
        1. For each product, consider these factors when deciding which single product to recommend:
              - Relevance to the user's request (explicit mentions like fabric, color)
              - Availability (prefer products not out_of_stock)
              - Value for money (selling_price, discount, actual_price) and rating
              - Important product attributes (fabric, color, and use the description to identify features)
        2. Identify the single best product that matches the user's request.
        3. Present the recommendation clearly in this format:
        - Best Product: [Product PID], [Product Name]

        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        4. If there is another product that could also work, mention it briefly as an alternative.
        5. If no product is a good fit, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """
    def _format_product_metadata(self, product, index: int = 1) -> str:
        """Return a compact metadata block with essential fields.
        """
        pid = product.pid
        title = product.title
        brand = product.brand or "Unknown"
        category = product.category or "N/A"
        selling_price = product.selling_price if product.selling_price is not None else "N/A"
        actual_price = product.actual_price if product.actual_price is not None else "N/A"
        discount = product.discount if product.discount is not None else 0
        rating = product.average_rating if product.average_rating is not None else "N/A"
        in_stock = "In Stock" if not product.out_of_stock else "Out of Stock"
        fabric = product.product_fabric or "Not specified"
        color = product.product_color or "Not specified"
        # include a short truncated description to give more context without using many tokens
        description = product.description or "No description"
        if len(description) > 150:
            description = description[:150].rsplit(' ', 1)[0] + "..."

        formatted = (
            f"{index}. [PID:{pid}] {title} | Brand: {brand} | Category: {category} | Price: {selling_price} | Orig: {actual_price} | "
            f"Discount: {discount}% | Rating: {rating} | Stock: {in_stock} | Fabric: {fabric} | Color: {color} | Desc: {description}"
        )
        return formatted

    def _parse_budget_from_query(self, query: str):
        """Extract a numeric budget from user's query if present, else None."""
        m = re.search(r"(?:under|below|less than|max|<=)\s*(\d{2,7})", query, flags=re.IGNORECASE)
        print(f"DEBUG _parse_budget_from_query: query='{query}', match={m}")
        if m:
            try:
                budget = float(m.group(1))
                print(f"DEBUG: Extracted budget: {budget}")
                return budget
            except Exception as e:
                print(f"DEBUG: Error parsing budget: {e}")
                return None
        print(f"DEBUG: No budget keyword found in query")
        return None

    def _detect_no_good_products(self, user_query: str, results: list, budget: float | None = None) -> bool:
        """
        Prefilter to decide if there are no suitable products before calling LLM.
        """
        if not results:
            return True

        #we consider that is not a good product if all are out of stock
        if all(getattr(r, "out_of_stock", False) for r in results):
            return True

        #Budget check: if user expressed a budget and NONE of the results are within budget with 10% margin, consider there is no good product.
        if budget is not None:
            any_within_budget = False
            for r in results:
                # Try selling_price first, then actual_price as fallback
                price = getattr(r, "selling_price", None) or getattr(r, "actual_price", None)
                try:
                    price_val = float(price) if price is not None else float('inf')
                except Exception:
                    price_val = float('inf')
                if price_val <= budget * 1.1:
                    any_within_budget = True
                    break
            if not any_within_budget:
                return True

        #Rating check on top 20
        #if there are rated items and none of them reach a minimum threshold, consider there are no good products
        RATING_THRESHOLD = 3.0
        top_n = results[:20]
        ratings = [r.average_rating for r in top_n if getattr(r, "average_rating", None) is not None]
        if ratings:
            if max(ratings) < RATING_THRESHOLD:
                return True

        return False

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 15) -> dict:
        """
        Generate a response using the retrieved search results. 
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # quick pre-filter: detect if there are no suitable products before calling the LLM
            budget = self._parse_budget_from_query(user_query)
            if self._detect_no_good_products(user_query, retrieved_results, budget):
                return "There are no good products that fit the request based on the retrieved results."

            # Limit context: pass only top-K candidates 
            top_k = min(max(1, int(top_N)), 17)
            candidates = retrieved_results[:top_k]

            # Format the retrieved results with compact metadata 
            formatted_results = "\n".join(
                [self._format_product_metadata(res, idx) for idx, res in enumerate(candidates, 1)]
            )

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER
