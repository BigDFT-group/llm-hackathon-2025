
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import nest_asyncio
from IPython import get_ipython
from IPython.core.display import HTML

from OntoFlow.agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from OntoFlow.agent.Onto_wa_rag.CONSTANT import (
    API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL,
    MAX_CONCURRENT, MAX_RESULTS, STORAGE_DIR
)
#from OntoFlow.agent.Onto_wa_rag.fortran_analysis.providers.consult import FortranEntityExplorer

from chat import Chat
# --- Imports IPython Magic ---
from IPython.core.magic import Magics, magics_class, line_cell_magic
from IPython.display import display, Markdown
from bigdft_notebook_assistant import BigDFTNotebookAssistant, HAS_PY3DMOL, HAS_NGLVIEW

# Appliquer nest_asyncio pour permettre l'utilisation d'asyncio dans un environnement déjà bouclé (comme Jupyter)
nest_asyncio.apply()


# ==============================================================================
# 1. FONCTIONS D'AFFICHAGE (HELPER FUNCTIONS)
# ==============================================================================

async def show_available_commands():
    """Show all available magic commands."""
    display(Markdown("""
### ✨ LARA - Available Magic Commands ✨

---

#### 🔍 **Search**
- **`<question>`**: (Without `/`) **Quick semantic search** for relevant content

---

#### 🧠 **Agent (Deep Analysis)**
- **`/agent <question>`**: **Full analysis** with the agentic rag (parser Jupyter)

---

#### ⚡ **Execution on HPC**
- **`/execute`**: Runs the last code snippet contained in the message on the HPC. Must be an unique fonction contain import

---

### 🎯 **When to Use Each Mode?**

| Mode | Use Case | Speed | Accuracy |
|------|----------|-------|----------|
| **Simple Search** (`query`) | Quick lookup of content | ⚡⚡⚡ | 🎯🎯 |
| **Unified Agent** (`/agent`) | Complex, multi-file analysis | ⚡ | 🎯🎯🎯🎯 |

"""))

async def display_query_result(result: Dict[str, Any]):
    """Affiche le résultat d'une query() standard."""
    display(Markdown(f"### 🤖 Réponse\n{result.get('answer', 'Pas de réponse')}"))
    sources = result.get('sources', [])
    if sources:
        md = "#### 📚 Sources\n"
        for source in sources:
            concepts = source.get('detected_concepts', [])
            concept_str = f"**Concepts**: {', '.join(concepts)}" if concepts else ""
            md += f"- **Fichier**: `{source['filename']}` | **Score**: {source['relevance_score']:.2f} | {concept_str}\n"
        display(Markdown(md))


async def display_hierarchical_result(result: Dict[str, Any]):
    """Affiche les résultats de la recherche hiérarchique."""
    display(Markdown(f"### 🤖 Réponse\n{result.get('answer', 'Pas de réponse')}"))
    hierarchical_results = result.get('hierarchical_results', {})
    if hierarchical_results:
        md = "#### 📊 Résultats par niveau conceptuel\n"
        for level, data in hierarchical_results.items():
            md += f"- **{data.get('display_name', level)}** ({len(data.get('results', []))} résultats):\n"
            for i, res in enumerate(data.get('results', [])[:3]):
                md += f"  - `{res['source_info'].get('filename')}` (sim: {res['similarity']:.2f})\n"
        display(Markdown(md))


async def display_document_list(docs: List[Dict[str, Any]]):
    """Affiche la liste des documents."""
    if not docs:
        display(Markdown("📁 Aucun document n'a été indexé."))
        return
    md = f"### 📁 {len(docs)} documents indexés\n"
    for doc in docs:
        md += f"- `{doc.get('filename', 'N/A')}` ({doc.get('total_chunks', 0)} chunks)\n"
    display(Markdown(md))


# ==============================================================================
# 2. CLASSE DE LA MAGIC IPYTHON
# ==============================================================================

@magics_class
class OntoRAGMagic(Magics):
    def __init__(self, shell):
        super(OntoRAGMagic, self).__init__(shell)
        self.rag = None
        self._initialized = False
        self.last_agent_response = None
        self.first_turn = True
        self.bigdft_assistant = None
        print("✨ OntoRAG Magic prêt. Initialisation au premier usage...")

    async def _initialize_rag(self):
        """Initialisation asynchrone du moteur RAG."""
        print("🚀 Initialisation du moteur OntoRAG (une seule fois)...")
        self.rag = OntoRAG(storage_dir=STORAGE_DIR, ontology_path=ONTOLOGY_PATH_TTL)
        await self.rag.initialize()
        # Initialiser l'assistant BigDFT
        self.bigdft_assistant = BigDFTNotebookAssistant(rag_system=self.rag)
        self._initialized = True

    async def _handle_agent_run(self, user_input: str):
        """Gère un tour de conversation avec l'agent unifié."""
        print("🧠 The agent thinks...")

        # ✅ UTILISER L'AGENT avec la version structurée
        agent_response = await self.rag.unified_agent.run(user_input, use_memory=True)

        if agent_response.status == "clarification_needed":
            question_from_agent = agent_response.clarification_question
            display(Markdown(f"""### ❓ The agent needs clarification
    > {question_from_agent}

    **To reply, use the command:** `%rag /agent_reply <your_response>`"""))

        elif agent_response.status == "success":
            # stockage de la dernière réponse
            self.last_agent_response = agent_response.answer
            # Affichage enrichi avec les métadonnées
            display(Markdown(f"### ✅ Final response from the agent\n{agent_response.answer}"))

            # Afficher les sources automatiquement
            if agent_response.sources_consulted:
                sources_md = "\n## 📚 Sources consulted :\n"
                for source in agent_response.sources_consulted:
                    sources_md += f"\n{source.get_citation()}"
                display(Markdown(sources_md))

            # Afficher les métadonnées utiles
            metadata_md = f"""
    ### 📊 Response metadata
    - ⏱️ **Execution time**: {agent_response.execution_time_total_ms:.0f}ms
    - 🔢 **Steps used**: {agent_response.steps_taken}/{agent_response.max_steps}
    - 📚 **Sources consultées**: {len(agent_response.sources_consulted)}
    - 🎯 **Niveau de confiance**: {agent_response.confidence_level:.2f}
    """

            # Ajouter les questions de suivi suggérées
            if agent_response.suggested_followup_queries:
                metadata_md += f"\n### 💡 Questions de suivi suggérées :\n"
                for i, suggestion in enumerate(agent_response.suggested_followup_queries[:3], 1):
                    metadata_md += f"{i}. {suggestion}\n"

            display(Markdown(metadata_md))
            print("\n✅ Conversation terminée. Pour une nouvelle question, utilisez à nouveau `/agent`.")

        elif agent_response.status == "timeout":
            display(Markdown(f"""### ⏰ Timeout de l'agent
    L'agent a atteint la limite de temps mais a trouvé des informations partielles :

    {agent_response.answer}"""))

            if agent_response.sources_consulted:
                sources_md = "\n## 📚 Sources consultées malgré le timeout :\n"
                for source in agent_response.sources_consulted:
                    sources_md += f"\n{source.get_citation()}"
                display(Markdown(sources_md))

        elif agent_response.status == "error":
            display(Markdown(f"""### ❌ Erreur de l'agent
    {agent_response.error_details}

    Essayez de reformuler votre question ou utilisez `/help` pour voir les commandes disponibles."""))

        else:
            display(Markdown(f"### ⚠️ Statut inattendu : {agent_response.status}"))

    async def _handle_simple_search(self, query: str, max_results: int = 5):
        """Effectue une recherche simple + génération de réponse avec le LLM."""
        print(f"🔍 Recherche simple RAG : '{query}'")

        # Vérifier si l'agent unifié est disponible
        if not hasattr(self.rag, 'unified_agent') or not self.rag.unified_agent:
            display(Markdown("❌ **Agent unifié non disponible**\n\nUtilisez `/search` pour la recherche classique."))
            return

        # 1. Effectuer la recherche sémantique
        #results = retriever.query(query, k=max_results)
        results = await self.rag.rag_engine.search(query=query, top_k=max_results)
        #results = self.rag.query(query=query, top_k=max_results)

        if not results:
            display(Markdown(f"""### 🔍 Recherche : "{query}"

    ❌ **Aucun résultat trouvé** (seuil de similarité : 0.25)

    **Suggestions :**
    - Essayez des termes plus généraux
    - `/agent {query}` pour une analyse approfondie  
    - `/search {query}` pour la recherche classique"""))
            return

        # 2. Générer la réponse avec le LLM
        print(f"  🤖 Génération de la réponse avec {len(results)} chunks de contexte...")

        try:
            answer, sources_info = await self._generate_rag_response(query, results)

            # 3. Afficher la réponse complète
            await self._display_rag_response(query, answer, results, sources_info)

        except Exception as e:
            print(f"❌ Erreur génération réponse: {e}")
            # Fallback : afficher les chunks bruts
            display(Markdown("⚠️ **Erreur de génération LLM**, affichage des chunks bruts :"))
            await self._display_simple_search_results(query, results)

    async def _generate_rag_response(self, query: str, results: List[Dict[str, Any]]) -> Tuple[
        str, List[Dict[str, Any]]]:
        """Génère une réponse avec le LLM à partir des chunks trouvés."""

        # 1. Construire le contexte depuis les chunks
        context_parts = []
        sources_info = []

        for i, result in enumerate(results, 1):
            source_filename = result.get("source_filename", "Unknown")
            content = result.get("content", "")
            similarity_score = result.get("similarity_score", 0.0)

            # Ajouter au contexte
            context_parts.append(f"[Source {i}] De {source_filename} (score: {similarity_score:.3f}):\n{content}")

            # Info pour les sources finales
            sources_info.append({
                "index": i,
                "filename": source_filename,
                "score": similarity_score,
                "source_file": result.get("source_file", ""),
                "tokens": result.get("tokens", "?")
            })

        context = "\n\n".join(context_parts)

        # 2. Construire le prompt pour le LLM
        system_prompt = """Tu es un assistant expert qui répond aux questions en citant TOUJOURS ses sources.

    Tu as accès aux sources suivantes provenant de notebooks Jupyter :
    - Cite OBLIGATOIREMENT tes sources en utilisant [Source N] dans ta réponse
    - Concentre-toi sur les informations les plus pertinentes
    - Structure ta réponse de manière claire et pratique
    - N'invente aucune information qui ne figure pas dans les sources

    Exemple de citation: "D'après [Source 1], pour créer une molécule..."
    """

        user_prompt = f"""Question: {query}

    Contexte disponible:
    {context}

    Réponds à la question en utilisant exclusivement les informations du contexte et en citant tes sources [Source N]."""

        # 3. Appeler le LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_response = await self.rag.rag_engine.llm_provider.generate_response(messages, temperature=0.3)

        return llm_response, sources_info

    async def _display_rag_response(self, query: str, answer: str, results: List[Dict], sources_info: List[Dict]):
        """Affiche la réponse RAG complète avec métadonnées."""

        # En-tête avec statistiques
        document_store = self.rag.rag_engine.document_store
        total_indexed = document_store.get_total_chunks_count()
        indexed_files_count = document_store.get_document_count()
        avg_score = sum(r.get("similarity_score", 0) for r in results) / len(results)

        header = f"""### 🤖 RAG answer: "{query}"

    📊 **Context: **{len(results)} chunks selected out of {total_indexed} available ({indexed_files_count} notebooks) | **Average score:** {avg_score:.3f}

    ---

    """

        # Corps de la réponse
        response_body = f"""### 💡 Answer

    {answer}

    ---

    """

        # Sources détaillées
        sources_section = "### 📚 Used sources\n\n"
        for source in sources_info:
            score_bar = "🟢" * int(source["score"] * 10) + "⚪" * (10 - int(source["score"] * 10))
            sources_section += f"""**[Source {source['index']}]** `{source['filename']}` | Score: {source['score']:.3f} {score_bar} | Tokens: {source['tokens']}\n\n"""

        # Actions suggérées
        footer = f"""
    ---

    ### 💡 Actions suggérées

    - **Analyse approfondie :** `%rag /agent {query}`
    - **Plus de contexte :** `%rag /simple_search_more {query}`  
    - **Voir les chunks bruts :** `%rag /simple_search_raw {query}`
    """

        # Afficher tout
        display(Markdown(header + response_body + sources_section))

    async def _display_simple_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Affiche les résultats de la recherche simple de manière attractive."""

        # En-tête avec statistiques
        document_store = self.rag.rag_engine.document_store
        total_indexed = document_store.get_total_chunks_count()
        indexed_files = document_store.get_document_count()

        header = f"""### 🔍 Search result : "{query}"

    📊 **{len(results)} result(s) found on {total_indexed} indexed chunks ({indexed_files} notebooks)

    ---
    """
        display(Markdown(header))

        # Afficher chaque résultat
        for i, result in enumerate(results, 1):
            score = result["similarity_score"]
            source_file = result["source_filename"]
            tokens = result.get("tokens", "?")
            content = result["content"]

            # Tronquer le contenu si trop long
            if len(content) > 800:
                content_preview = content[:800] + "\n\n*[...truncated content...]*"
            else:
                content_preview = content

            # Barre de score visuelle
            score_bar = "🟢" * int(score * 10) + "⚪" * (10 - int(score * 10))

            result_md = f"""
    #### 📄 Result {i}/{len(results)}

    **📁 Source :** `{source_file}` | **🎯 Score :** {score:.3f} {score_bar} | **📝 Tokens :** {tokens}

    ```
    {content_preview}
    ```

    ---
    """
            display(Markdown(result_md))

    async def _handle_execute(self):
        """Récupère le code de la réponse structurée et l'exécute sur le HPC."""
        try:
            # 1. Vérifier qu'on a une réponse d'agent récente
            if not self.last_agent_response:
                display(Markdown(
                    "❌ **No recent agent responses found**\n\nFirst use `/agent <question>` then `/execute`"))
                return

            # 2. Vérifier qu'on a une réponse structurée avec du code
            if (not hasattr(self.last_agent_response, 'structured_answer') or
                    not self.last_agent_response.structured_answer or
                    not self.last_agent_response.structured_answer.code_examples):
                display(
                    Markdown("❌ **No code found in the response**\n\nThe agent did not provide a code example."))
                return

            # 3. Récupérer le premier exemple de code Python
            code_examples = self.last_agent_response.structured_answer.code_examples
            python_code = None
            explanation = None
            for example in code_examples:
                if example.language.lower() in ['python', 'py']:
                    python_code = example.code
                    explanation = example.explanation
                    break

            if not python_code:
                display(Markdown("❌ **No Python code found**\n\nThe examples are not in Python."))
                return

            # 4. Afficher et exécuter
            display(Markdown(f"### 🚀 Starting the HPC agent..."))
            print(f"📝 Execution: {explanation}")

            chat = Chat(model="gpt-5")
            message = f"""
    Use the remote_run_code tool to run the following python function on 'robin-ubuntu':
    function_source='{python_code}', with function_args={{}}
    """

            await chat.chat(message)
            chat.print_history()

        except Exception as e:
            print(f"❌ Error while executing code: {e}")
            import traceback
            traceback.print_exc()

    async def _display_bigdft_result(self, result: Dict[str, Any]):
        """Affiche les résultats de l'assistant BigDFT de manière structurée."""

        status = result.get('status', 'unknown')

        # 🎯 Dispatch vers la méthode d'affichage appropriée
        display_methods = {
            # Démarrage et bienvenue
            'started': self._display_welcome,

            # Système moléculaire
            'system_created': self._display_system_created,
            'need_coordinates': self._display_coordinate_request,
            'need_more_info': self._display_info_request,
            'system_proposed_structured': self._display_system_proposed_structured,

            # Configuration calcul
            'code_ready': self._display_code_ready,
            'configuration_updated': self._display_config_updated,
            'need_clarification': self._display_clarification,

            # Exécution
            'execution_help': self._display_execution_help,
            'config_display': self._display_current_config,
            'code_regenerated': self._display_code_regenerated,

            # Réponses et aide
            'rag_response': self._display_rag_response,
            'ready': self._display_ready_status,

            'error': self._display_error,
            'no_changes': self._display_no_changes,
            'plan_awaits_confirmation': self._display_pass_through,
            'final_analysis_ready': self._display_final_analysis_result,

            'unknown': self._display_unknown_status
        }

        # Exécuter la méthode d'affichage appropriée
        display_method = display_methods.get(status, self._display_unknown_status)
        await display_method(result)

    # ============================================================================
    # MÉTHODES D'AFFICHAGE SPÉCIALISÉES
    # ============================================================================

    async def _display_pass_through(self, result: Dict[str, Any]):
        """
        Une méthode d'affichage asynchrone qui ne fait rien.
        Utilisée pour les statuts qui ont déjà géré leur propre affichage en amont,
        comme la confirmation d'un plan.
        """
        pass

    async def _display_welcome(self, result: Dict[str, Any]):
        """Affiche le message de bienvenue."""
        message = result.get('message', '')
        display(Markdown(message))

    async def _display_system_proposed_structured(self, result: Dict[str, Any]):
        """Affiche une réponse structurée BigDFT et exécute directement la visualisation."""

        structured_response = result.get('structured_response')
        if not structured_response:
            return await self._display_error({"message": "Réponse structurée manquante"})

        # Résumé exécutif
        display(Markdown(f"### ✅ {structured_response.executive_summary}"))

        # Informations sur la molécule
        if structured_response.molecule_proposal:
            mol = structured_response.molecule_proposal
            molecule_md = f"""
    ### 🧪 Molécule proposée : {mol.name}
    - **Atomes :** {len(mol.atoms)}
    - **Charge :** {mol.charge}  
    - **Multiplicité :** {mol.multiplicity}
    - **Confiance :** {mol.confidence:.1%}
    - **Géométrie :** {mol.geometry_type}

    **Explication :** {mol.explanation}
    """
            display(Markdown(molecule_md))

        # ✅ SOLUTION SIMPLE : Créer et exécuter directement
        if structured_response.visualization_code:
            await self._create_and_run_visualization(structured_response.molecule_proposal)

        # Instructions suivantes
        display(Markdown(f"### ➡️ Prochaines étapes\n{structured_response.next_instructions}"))

    async def _create_and_run_visualization(self, molecule_proposal):
        """Crée une nouvelle cellule de notebook avec le code de visualisation pour que l'utilisateur l'exécute."""

        try:
            # 1. Extraire les données de la molécule (identique)
            molecule_data = {
                "name": molecule_proposal.name,
                "charge": molecule_proposal.charge,
                "multiplicity": molecule_proposal.multiplicity,
                "atoms": [
                    {"element": atom.element, "position": atom.position}
                    for atom in molecule_proposal.atoms
                ]
            }

            # 2. Générer le template de code de visualisation (identique)
            visualization_code = f"""
# --- Cellule générée par l'Assistant BigDFT ---
# 🧪 Visualisation de {molecule_data['name']}
# ✏️ Modifiez les coordonnées ci-dessous puis ré-exécutez cette cellule (Shift+Entrée).

# Données moléculaires modifiables
molecule_data = {{
    "name": "{molecule_data['name']}",
    "charge": {molecule_data['charge']},
    "multiplicity": {molecule_data['multiplicity']},
    "atoms": ["""

            for atom in molecule_data["atoms"]:
                pos = atom["position"]
                visualization_code += f"""
        {{"element": "{atom['element']}", "position": [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]}},"""

            visualization_code += f"""
    ]
}}

# Génération du format XYZ pour la visualisation
def generate_xyz():
    lines = [str(len(molecule_data["atoms"]))]
    lines.append(f"{{molecule_data['name']}} - Modifiable par l'utilisateur")
    for atom in molecule_data["atoms"]:
        pos = atom["position"]
        lines.append(f"{{atom['element']:2s}} {{pos[0]:12.6f}} {{pos[1]:12.6f}} {{pos[2]:12.6f}}")
    return "\\n".join(lines)

xyz_content = generate_xyz()

# Visualisation 3D avec py3Dmol
try:
    import py3Dmol
    print("🧪 Génération de la visualisation 3D interactive...")

    view = py3Dmol.view(width=700, height=500)
    view.addModel(xyz_content, 'xyz')
    view.setStyle({{'stick': {{'radius': 0.15}}, 'sphere': {{'radius': 0.4}}}})
    view.setBackgroundColor('#f8f9fa')
    view.zoomTo()
    view.show()

    print(f"✅ Molécule {{molecule_data['name']}} visualisée !")
    print(f"⚛️  {{len(molecule_data['atoms'])}} atomes | Charge: {{molecule_data['charge']}}")
    print("\\n💡 Pour continuer, tapez '%rag /discuss ok' dans la cellule magique suivante.")

except ImportError:
    print("⚠️ py3Dmol n'est pas installé. Installation : pip install py3Dmol")
    print("📊 Structure au format XYZ :")
    print(xyz_content)
"""

            # --- PARTIE CORRIGÉE ---
            # 3. Utiliser l'API IPython pour créer la cellule SANS l'exécuter
            ipython = get_ipython()
            if not ipython:
                raise RuntimeError("L'environnement IPython est introuvable. Impossible de créer la cellule.")

            # Message d'instruction clair pour l'utilisateur
            display(Markdown(
                "### 📝 Cellule de visualisation créée ! 👇\n\n"
                "**Veuillez exécuter la nouvelle cellule qui vient d'apparaître ci-dessous (en cliquant dessus puis `Shift+Entrée`) pour afficher la molécule.**"
            ))

            # Crée la nouvelle cellule et y insère le code.
            ipython.set_next_input(visualization_code)

        except Exception as e:
            print(f"❌ Erreur critique lors de la création de la cellule : {e}")
            # Le fallback est toujours utile en cas de problème
            await self._fallback_display_code(visualization_code,
                                              "Code de visualisation de la molécule")

    async def _fallback_display_code(self, code: str, description: str):
        """Fallback : affiche le code de manière copiable si JavaScript échoue."""

        display(Markdown(f"### ⚠️ Création automatique échouée"))
        display(Markdown(f"**{description}**"))
        display(Markdown("**Copiez ce code dans une nouvelle cellule :**"))

        # Utiliser un bloc de code avec bouton de copie
        display(Markdown(f"```python\n{code}\n```"))

        # Instructions claires
        display(Markdown("""
    ### 📋 Instructions :
    1. **Créez une nouvelle cellule** (bouton + ou Insert > Cell Below)
    2. **Copiez le code ci-dessus** 
    3. **Collez dans la nouvelle cellule**
    4. **Exécutez la cellule** (Shift+Enter)
    5. **Modifiez les coordonnées** si nécessaire et ré-exécutez
    """))

    async def _display_system_created(self, result: Dict[str, Any]):
        """Affiche la confirmation de création du système avec visualisation."""
        message = result.get('message', 'Système créé !')

        display(Markdown(f"### ✅ {message}"))

        # Informations du système
        system_info = result.get('system_info', {})
        if system_info:
            info_md = f"""
    ### 🧪 Système moléculaire
    - **Nom :** {system_info.get('name', 'N/A')}
    - **Nombre d'atomes :** {len(system_info.get('atoms', []))}
    - **Charge :** {system_info.get('charge', 0)}
    - **Multiplicité :** {system_info.get('multiplicity', 1)}
    """
            display(Markdown(info_md))

        # Visualisation 3D
        visualization = result.get('visualization')
        if visualization:
            display(Markdown("### 🔬 Structure 3D"))
            await self._display_3d_structure(visualization)

        display(Markdown("### ➡️ **Prochaine étape :** Configuration du calcul DFT"))

    async def _display_coordinate_request(self, result: Dict[str, Any]):
        """Affiche une demande de coordonnées atomiques."""
        message = result.get('message', '')

        display(Markdown(f"### 📝 {message}"))

        example = result.get('example', '')
        if example:
            display(Markdown(f"**Exemple de format :**\n```\n{example}\n```"))

        suggestions = result.get('suggestions', [])
        if suggestions:
            suggestions_md = "### 💡 Format attendu :\n" + "\n".join([f"- {s}" for s in suggestions])
            display(Markdown(suggestions_md))

    async def _display_info_request(self, result: Dict[str, Any]):
        """Affiche une demande d'informations supplémentaires."""
        message = result.get('message', '')

        display(Markdown(f"### 🤔 {message}"))

        suggestions = result.get('suggestions', [])
        if suggestions:
            suggestions_md = "### 💡 Suggestions :\n" + "\n".join([f"- {s}" for s in suggestions])
            display(Markdown(suggestions_md))

    async def _display_code_ready(self, result: Dict[str, Any]):
        """Affiche la confirmation que le code est prêt ET AFFICHE LE CODE LUI-MEME."""
        message = result.get('message', 'Code prêt !')

        display(Markdown(f"### ✅ {message}"))

        # Configuration résumée (inchangée)
        config = result.get('config_summary', {})
        if config:
            config_md = f"""
    ### ⚙️ Configuration du calcul
    - **Fonctionnelle DFT :** {config.get('functional', 'PBE')}
    - **Base atomique :** {config.get('basis_set', 'SZ')}
    - **Optimisation géométrie :** {'✅ Activée' if config.get('optimize', False) else '❌ Désactivée'}
    """
            display(Markdown(config_md))

        # --- AJOUT IMPORTANT : AFFICHER LE CODE GÉNÉRÉ ---
        bigdft_code = result.get('code')
        if bigdft_code:
            display(Markdown("### 📄 Code PyBigDFT généré"))
            display(Markdown(
                "Voici le code qui sera envoyé au HPC lorsque vous utiliserez la commande `/execute`."
            ))
            # On affiche le code dans un bloc python formaté
            display(Markdown(f"```python\n{bigdft_code}\n```"))
        # ----------------------------------------------------

        # Préparation pour l'exécution (inchangée)
        display(Markdown("""
    ### 🚀 Prêt pour l'exécution !

    Vous pouvez maintenant lancer ce code sur le HPC.

    **Pour lancer le calcul :**
    ```
    %rag /execute
    ```

    **Autres actions disponibles :**
    - `/discuss_status` : Voir l'état de la session
    - `/discuss <modification>` : Modifier la configuration (ex: `/discuss use B3LYP`)
    """))

        # Stocker le code pour /execute (inchangée)
        await self._prepare_code_for_execution(result)

    async def _display_config_updated(self, result: Dict[str, Any]):
        """Affiche les modifications de configuration."""
        message = result.get('message', 'Configuration mise à jour !')

        display(Markdown(f"### ✅ {message}"))

        # Changements apportés
        changes = result.get('changes', [])
        if changes:
            changes_md = "### 🔧 Modifications apportées :\n" + "\n".join([f"- ✨ {c}" for c in changes])
            display(Markdown(changes_md))

        # Configuration finale
        config = result.get('config_summary', {})
        if config:
            config_md = f"""
    ### ⚙️ Configuration finale
    - **Fonctionnelle :** {config.get('functional', 'N/A')}
    - **Base :** {config.get('basis_set', 'N/A')}
    - **Optimisation :** {'✅' if config.get('optimize', False) else '❌'}
    - **Spin polarisé :** {'✅' if config.get('spin_polarized', False) else '❌'}
    """
            display(Markdown(config_md))

        display(Markdown("### 🚀 **Utilisez `/execute` pour lancer le calcul**"))

        # Stocker le code pour /execute
        await self._prepare_code_for_execution(result)

    async def _display_clarification(self, result: Dict[str, Any]):
        """Affiche une demande de clarification."""
        message = result.get('message', '')

        display(Markdown(f"### 🤔 Clarification nécessaire\n{message}"))

        suggestions = result.get('suggestions', [])
        if suggestions:
            suggestions_md = "### 💡 Suggestions :\n" + "\n".join([f"- {s}" for s in suggestions])
            display(Markdown(suggestions_md))

    async def _display_execution_help(self, result: Dict[str, Any]):
        """Affiche l'aide pour l'exécution."""
        message = result.get('message', '')
        display(Markdown(message))

        if result.get('ready_for_execute', False):
            display(Markdown("### 🎯 **Action recommandée :** `%rag /execute`"))

    async def _display_current_config(self, result: Dict[str, Any]):
        """Affiche la configuration actuelle détaillée."""
        display(Markdown("### ⚙️ Configuration actuelle de la simulation BigDFT"))

        config = result.get('config', {})

        # Informations système
        system_info = config.get('system', {})
        system_md = f"""
    #### 🧪 Système moléculaire
    - **Nom :** {system_info.get('name', 'Non défini')}
    - **Nombre d'atomes :** {system_info.get('atoms', 0)}
    - **Charge totale :** {system_info.get('charge', 0)}
    - **Multiplicité :** {system_info.get('multiplicity', 1)}
    """
        display(Markdown(system_md))

        # Informations calcul
        calc_info = config.get('calculation', {})
        if calc_info:
            calc_md = f"""
    #### ⚛️ Paramètres DFT
    - **Fonctionnelle :** {calc_info.get('functional', 'N/A')}
    - **Base atomique :** {calc_info.get('basis_set', 'N/A')}
    - **Optimisation géométrie :** {'✅ Activée' if calc_info.get('optimize', False) else '❌ Désactivée'}
    - **Calcul polarisé en spin :** {'✅ Activé' if calc_info.get('spin_polarized', False) else '❌ Désactivé'}
    """
        else:
            calc_md = "#### ⚛️ Paramètres DFT : **Non configurés**"

        display(Markdown(calc_md))

    async def _display_code_regenerated(self, result: Dict[str, Any]):
        """Affiche la confirmation de régénération du code."""
        message = result.get('message', 'Code régénéré !')

        display(Markdown(f"### ✅ {message}"))
        display(Markdown("Le code PyBigDFT a été mis à jour avec la configuration actuelle."))
        display(Markdown("### 🚀 **Utilisez `/execute` pour lancer le calcul**"))

        # Stocker le code pour /execute
        await self._prepare_code_for_execution(result)

    async def _display_ready_status(self, result: Dict[str, Any]):
        """Affiche le statut prêt pour l'exécution."""
        message = result.get('message', 'Calcul prêt !')

        display(Markdown(f"### 🎯 {message}"))
        display(Markdown("""
    **Actions disponibles :**
    - `%rag /execute` : Lancer le calcul sur le HPC
    - `%rag /discuss_config` : Voir la configuration
    - `%rag /discuss <question>` : Poser une question
    """))

    async def _display_no_changes(self, result: Dict[str, Any]):
        """Affiche quand aucun changement n'a été détecté."""
        message = result.get('message', '')

        display(Markdown(f"### 🤷‍♂️ {message}"))

        current_config = result.get('current_config', {})
        if current_config:
            config_md = f"""
    ### 📋 Configuration actuelle :
    - **Fonctionnelle :** {current_config.get('functional', 'N/A')}
    - **Base :** {current_config.get('basis_set', 'N/A')}
    - **Optimisation :** {'✅' if current_config.get('optimize', False) else '❌'}
    """
            display(Markdown(config_md))

        suggestions = result.get('suggestions', [])
        if suggestions:
            suggestions_md = "### 💡 Modifications possibles :\n" + "\n".join([f"- {s}" for s in suggestions])
            display(Markdown(suggestions_md))

    async def _display_error(self, result: Dict[str, Any]):
        """Affiche les erreurs."""
        message = result.get('message', 'Une erreur est survenue')

        display(Markdown(f"### ❌ Erreur\n{message}"))

        # Suggestions pour résoudre l'erreur
        display(Markdown("""
    ### 🔧 Solutions possibles :
    - Vérifiez votre demande et reformulez si nécessaire
    - Utilisez `/discuss_reset` pour recommencer
    - Consultez `/help` pour les commandes disponibles
    """))

    async def _display_unknown_status(self, result: Dict[str, Any]):
        """Affiche un statut inconnu."""
        status = result.get('status', 'unknown')
        message = result.get('message', 'Statut inconnu')

        display(Markdown(f"### ⚠️ Statut non géré : `{status}`\n{message}"))

    # ============================================================================
    # MÉTHODE UTILITAIRE POUR L'EXÉCUTION
    # ============================================================================

    async def _prepare_code_for_execution(self, result: Dict[str, Any]):
        """Prépare le code BigDFT pour l'exécution via /execute."""
        bigdft_code = result.get('code')

        if bigdft_code and self.bigdft_assistant:
            # Créer une réponse structurée pour /execute
            from OntoFlow.agent.Onto_wa_rag.jupyter_analysis.jupyter_agent import AgentStructuredAnswerArgs, CodeExample
            from OntoFlow.agent.Onto_wa_rag.jupyter_analysis.jupyter_agent import AgentResponse
            from datetime import datetime

            structured_response = AgentStructuredAnswerArgs(
                executive_summary="Code PyBigDFT généré par l'assistant et prêt pour l'exécution sur HPC.",
                code_examples=[
                    CodeExample(
                        language="python",
                        code=bigdft_code,
                        explanation="Calcul BigDFT complet avec PyBigDFT",
                        function_name="run_bigdft_calculation",
                        execution_ready=True,
                        is_complete_function=True,
                        required_modules=["BigDFT", "numpy"]
                    )
                ],
                answer_type="hpc_function"
            )

            # Créer une réponse d'agent simulée pour /execute
            fake_response = AgentResponse(
                answer="Code BigDFT généré par l'assistant",
                status="success",
                query="BigDFT calculation generation",
                session_id="bigdft_session",
                timestamp=datetime.now(),
                execution_time_total_ms=1000.0,
                steps_taken=1,
                max_steps=1,
                structured_answer=structured_response
            )

            # Stocker pour /execute
            self.last_agent_response = fake_response

            print("💾 Code BigDFT préparé pour /execute")

    async def _display_final_analysis_result(self, result: Dict[str, Any]):
        """Affiche le résultat formaté de l'analyse finale du plan."""
        md = f"""
    ### 📊 Analyse Finale : {result.get('description')}

    Le calcul a été effectué en utilisant la formule :
    `{result.get('formula')}`

    Avec les valeurs obtenues :
    `{result.get('readable_formula')}`

    ---
    ### 🎉 Résultat Final : {result.get('final_result'):.4f} [Ha]

    Le plan est terminé. Vous pouvez lancer une nouvelle discussion.
    """
        display(Markdown(md))

        # On peut réinitialiser l'assistant ici si besoin
        if self.bigdft_assistant:
            await self.bigdft_assistant.start_discussion()

    async def _display_3d_structure(self, visualization: Dict[str, Any]):
        """Affiche une structure 3D dans le notebook."""

        viz_type = visualization.get('type', 'text')
        data = visualization.get('data', '')

        if viz_type == 'py3dmol':
            try:
                # ✅ Import local pour éviter les erreurs PyCharm
                import py3Dmol

                view = py3Dmol.view(width=600, height=400)
                view.addModel(data, 'xyz')
                view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'radius': 0.5}})
                view.setBackgroundColor('white')
                view.zoomTo()

                display(HTML(view._make_html()))

            except ImportError:
                display(Markdown("❌ **py3Dmol non installé**"))
                display(Markdown("📦 **Installation :** `pip install py3Dmol`"))
                # Fallback vers affichage texte
                display(Markdown("### 🔬 Structure (format XYZ)"))
                display(Markdown(f"```\n{data}\n```"))

        else:
            # Fallback : affichage texte
            display(Markdown("### 🔬 Structure (format XYZ)"))
            display(Markdown(f"```\n{data}\n```"))

    @line_cell_magic
    def rag(self, line, cell=None):
        """Magic command principale pour interagir avec OntoRAG."""
        async def main():
            if self.first_turn:
                await show_available_commands()
                self.first_turn = False
            if not self._initialized:
                await self._initialize_rag()

            query = cell.strip() if cell else line.strip()
            if not query:
                await show_available_commands()
                return

            parts = query.split(' ', 1)
            command = parts[0]
            args = parts[1].strip() if len(parts) > 1 else ""

            try:
                if command.startswith('/'):
                    # --- COMMANDES DE L'AGENT UNIFIÉ ---
                    if command == '/agent':
                        print("💫 Command : /agent", args)
                        await self._handle_agent_run(args)

                    elif command == '/agent_reply':
                        print("💬 User response :", args)
                        await self._handle_agent_run(args)

                    elif command == '/agent_memory':
                        """Affiche le résumé de la mémoire de l'agent."""
                        memory_summary = self.rag.unified_agent.get_memory_summary()
                        display(Markdown(f"### 🧠 Agent memory\n{memory_summary}"))

                    elif command == '/agent_clear':
                        """Efface la mémoire de l'agent."""
                        self.rag.unified_agent.clear_memory()
                        display(Markdown("### 🧹 Agent memory cleared"))

                    elif command == '/agent_sources':
                        """Affiche toutes les sources utilisées dans la session."""
                        sources = self.rag.unified_agent.get_sources_used()
                        if sources:
                            sources_md = f"### 📚 Session sources ({len(sources)} references)\n"
                            for source in sources:
                                sources_md += f"\n{source.get_citation()}"
                            display(Markdown(sources_md))
                        else:
                            display(Markdown("### 📚 No sources consulted in this session"))

                    elif command == '/add_docs':
                        var_name = args.strip()
                        documents_to_add = self.shell.user_ns.get(var_name)
                        if documents_to_add is None:
                            print(f"❌ Variable '{var_name}' not found.")
                            return
                        print(f"📚 Adding {len(documents_to_add)} documents...")
                        results = await self.rag.add_documents_batch(documents_to_add, max_concurrent=MAX_CONCURRENT)
                        print(f"✅ Addition complete: {sum(results.values())}/{len(results)} successes.")

                    elif command == '/list':
                        docs = self.rag.list_documents()
                        await display_document_list(docs)

                    elif command == '/stats':
                        stats = self.rag.get_statistics()
                        display(Markdown(f"### 📊 Statistic OntoRAG\n```json\n{json.dumps(stats, indent=2)}\n```"))

                    elif command == '/search':
                        result = await self.rag.query(args, max_results=MAX_RESULTS)
                        await display_query_result(result)

                    elif command == '/hierarchical':
                        result = await self.rag.hierarchical_query(args)
                        await display_hierarchical_result(result)

                    elif command == '/help':
                        await show_available_commands()

                    elif command == '/discuss':
                        """Démarre une discussion BigDFT interactive."""
                        if not self.bigdft_assistant:
                            display(Markdown("❌ **Assistant BigDFT non initialisé**"))
                            return

                        if args.strip():
                            # Continuer une discussion
                            result = await self.bigdft_assistant.process_message(args)
                        else:
                            # Nouvelle discussion
                            result = await self.bigdft_assistant.start_discussion()

                        await self._display_bigdft_result(result)

                    elif command == '/discuss_status':
                        """Affiche l'état de la discussion BigDFT."""
                        if not self.bigdft_assistant:
                            display(Markdown("❌ **Assistant BigDFT non initialisé**"))
                            return

                        state = self.bigdft_assistant.get_session_state()
                        display(Markdown(f"""### 🔬 État de la Session BigDFT

                    **Étape actuelle :** {state['stage']}
                    **Système défini :** {'✅' if state['has_system'] else '❌'}
                    **Calcul configuré :** {'✅' if state['has_calculator'] else '❌'}
                    **Messages échangés :** {state['conversation_length']}
                    """))

                    elif command == '/discuss_reset':
                        """Remet à zéro la discussion BigDFT."""
                        if self.bigdft_assistant:
                            result = await self.bigdft_assistant.start_discussion()
                            display(Markdown("🔄 **Discussion BigDFT réinitialisée**"))
                            await self._display_bigdft_result(result)

                    elif command == '/execute':
                        print("🚀 Extracting and executing code from the last comment cell...")
                        await self._handle_execute()

                    else:
                        print(f"❌ Unknow command: '{command}'.")
                        await show_available_commands()

                else:  # Requête en langage naturel directe
                    print("🤖 Direct request via SimpleRetriever...")
                    await self._handle_simple_search(query, max_results=5)

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

        asyncio.run(main())



# ==============================================================================
# 3. FONCTION DE CHARGEMENT IPYTHON
# ==============================================================================

def load_ipython_extension(ipython):
    """Enregistre la classe de magics lors du chargement de l'extension."""
    ipython.register_magics(OntoRAGMagic)