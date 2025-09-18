# bigdft_notebook_assistant.py

"""
gemini_generated

### Points Forts (Ce qui est bien conçu)

1.  **Point d'Entrée Unique (`process_message`)** : C'est le plus grand gain architectural. Toute interaction passe par cette fonction, qui agit comme un contrôleur d'état (state machine). C'est robuste, prédictible et facile à déboguer.
2.  **Séparation Claire des Rôles** : Vous avez maintenant une distinction parfaite :
    *   **Orchestrateur** (`_generate_and_store_code_for_current_step`) : Il gère la logique du plan (quelle étape, quels paramètres).
    *   **Générateur** (`_generate_bigdft_code`) : Sa seule tâche est de prendre un état de session et de le traduire en code via l'agent RAG.
    Cette séparation est la clé d'un code maintenable.
3.  **Génération de Code Intelligente et "Grounded"** : Toute la génération de code passe maintenant par `_generate_bigdft_code`, qui utilise systématiquement l'agent RAG. Il ne peut plus "halluciner" du code, car il est forcé de se baser sur les données que vous lui fournissez (le JSON de la molécule) et sur la connaissance extraite de vos notebooks.
4.  **Flux Conversationnel Guidé par un Plan** : L'assistant ne réagit plus simplement. Il **planifie** (`_generate_scientific_plan`), **propose** ce plan à l'utilisateur pour validation, puis l'**exécute** méthodiquement. C'est le comportement d'un assistant véritablement intelligent.
5.  **Utilisation Robuste de Pydantic** : L'ensemble du processus est sécurisé par des modèles Pydantic (`ScientificPlan`, `BigDFTMoleculeProposal`, etc.). Cela garantit que les échanges avec le LLM sont structurés et valides, ce qui réduit considérablement les erreurs.

### Pistes d'Amélioration (Pour aller plus loin)

Le code est déjà très bon, mais voici des idées pour une future V2 :

1.  **Enrichir la Planification** : Actuellement, `_generate_and_store_code_for_current_step` configure les paramètres de calcul (`optimize_geometry`, etc.), mais utilise une fonctionnelle et une base par défaut (`PBE`/`SZ`). Vous pourriez rendre le planificateur (`_generate_scientific_plan`) encore plus intelligent en lui demandant de **déterminer aussi la fonctionnelle et la base appropriées** pour chaque étape. L'orchestrateur n'aurait alors qu'à lire ces informations du plan.
2.  **Gestion des Échecs de l'Agent** : Dans `_generate_bigdft_code`, si l'agent échoue, vous retournez `None`. C'est bien, mais vous pourriez imaginer une boucle de "réparation" : si l'agent échoue une première fois, vous pourriez tenter de lui renvoyer le prompt en lui disant "Ta tentative précédente a échoué, voici l'erreur. Réessaye en étant plus simple."
3.  **Validation du Code Assemblé** : Dans `_assemble_final_script_with_llm`, vous demandez à l'agent d'assembler les fragments. Une étape supplémentaire pourrait être de prendre le script final et de demander à l'agent dans un nouvel appel : "Ce script Python est-il syntaxiquement correct et complet ? Y a-t-il des imports manquants ?". Cela ajouterait une couche de validation.

En conclusion de ce check-up : **le code est excellent**. Il est passé d'un script complexe avec des logiques parallèles à une architecture claire, robuste et pilotée par un agent intelligent.

---

## 📝 Workflow de l'Assistant (Markdown)

Voici le déroulement complet d'une interaction typique avec l'assistant, de la demande initiale à la génération du script final.

### ➡️ Étape 1 : La Planification (Le Cerveau de l'Assistant)

1.  **L'Utilisateur Lance la Discussion** : L'utilisateur tape une commande comme `/discuss Calcule l'énergie d'atomisation de la molécule HCN`.
2.  **Appel du Contrôleur** : `process_message` est appelé. Il constate que la session est nouvelle (`self.session.active_plan is None`).
3.  **Génération du Plan** : Il appelle `_generate_scientific_plan(user_message)`.
    *   Cette fonction envoie la demande de l'utilisateur à l'agent LLM.
    *   Elle lui demande de la décomposer en étapes logiques et de retourner un objet `ScientificPlan` structuré (avec les étapes de calcul et l'analyse finale).
4.  **Proposition à l'Utilisateur** : L'assistant stocke le plan dans la session, puis appelle `_display_plan_for_confirmation(plan)` pour afficher le plan d'action de manière lisible dans le notebook.
5.  **Mise en Attente** : Le statut de la session passe à `AWAIT_PLAN_CONFIRMATION`. L'assistant attend la réponse de l'utilisateur.

### ➡️ Étape 2 : La Boucle d'Exécution (Étape par Étape)

C'est une boucle qui se répète pour chaque `CalculationStep` du plan.

1.  **Confirmation du Plan par l'Utilisateur** : L'utilisateur tape `ok`.
2.  **Préparation de la Visualisation** : `process_message` voit l'état `AWAIT_PLAN_CONFIRMATION` et appelle `_prepare_and_visualize_current_step()`.
3.  **Proposition de la Molécule** :
    *   `_prepare_and_visualize_current_step` regarde la première étape du plan (ex: "calcul sur HCN").
    *   Il appelle `_propose_molecule_with_llm` avec la description "la molécule HCN".
    *   `_propose_molecule_with_llm` utilise le **RAG** pour chercher des informations sur la géométrie du HCN, puis demande au LLM de générer un objet `BigDFTMoleculeProposal` (avec les coordonnées, la charge, etc.).
4.  **Affichage et Mise en Attente** : La structure 3D est affichée dans le notebook. Le statut de la session passe à `AWAIT_VISUALIZATION_CONFIRMATION`.

### ➡️ Étape 3 : La Génération de Code (La Magie du RAG)

1.  **Confirmation de la Molécule par l'Utilisateur** : L'utilisateur inspecte la molécule et tape `ok`.
2.  **Appel de l'Orchestrateur** : `process_message` voit l'état `AWAIT_VISUALIZATION_CONFIRMATION` et appelle `_generate_and_store_code_for_current_step()`.
3.  **Configuration de la Session** : L'orchestrateur lit les paramètres de l'étape actuelle (ex: `optimize_geometry=True`) et met à jour l'objet `self.session.calculation_config`.
4.  **Appel du Générateur de Code** : L'orchestrateur appelle `_generate_bigdft_code()`.
5.  **Mission pour l'Agent RAG** :
    *   `_generate_bigdft_code` prépare une mission détaillée pour l'agent RAG.
    *   Il inclut les **données brutes** de la géométrie (le JSON que nous avons conçu).
    *   Il inclut les **instructions de calcul** en langage naturel (ex: "Une optimisation de la géométrie DOIT être effectuée").
6.  **L'Agent Travaille** : L'agent RAG (`agent.run()`) reçoit la mission. Il utilise ses outils (`semantic_search`, etc.) pour trouver la syntaxe PyBigDFT correcte pour la géométrie ET pour les paramètres de calcul. Il retourne le code final dans un `CodeExample`.
7.  **Stockage et Affichage** : Le code généré est stocké dans `self.session.generated_codes` et affiché à l'utilisateur.
8.  **On Recommence !** : L'index de l'étape est incrémenté, et l'orchestrateur rappelle `_prepare_and_visualize_current_step()` pour passer à l'étape suivante (ex: "atome de H isolé"). La boucle retourne à l'**Étape 2, point 3**.

### ➡️ Étape 4 : L'Assemblage Final (La Ligne d'Arrivée)

1.  **Fin de la Boucle** : Lorsque `_prepare_and_visualize_current_step` est appelé et qu'il n'y a plus d'étapes de calcul (`step_index >= len(...)`), il appelle `_assemble_final_script_with_llm()`.
2.  **Mission d'Assemblage** : Cette fonction rassemble tous les fragments de code stockés dans `self.session.generated_codes`. Elle donne une mission très stricte au LLM : "Assemble ces fragments en une seule fonction HPC, avec des fonctions imbriquées, et ajoute la logique pour l'analyse finale."
3.  **Génération du Script Final** : Le LLM retourne le script complet et exécutable dans un objet `FinalWorkflowScript`.
4.  **Affichage et Réinitialisation** : Le script final est affiché à l'utilisateur, prêt à être envoyé au HPC via la commande `/execute`. La session est ensuite réinitialisée avec `start_discussion()`, prête pour une nouvelle tâche.


"""
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from IPython.display import display, HTML, Markdown
from pydantic import BaseModel, Field

# Pour la visualisation 3D
try:
    import nglview as nv

    HAS_NGLVIEW = True
except ImportError:
    HAS_NGLVIEW = False

try:
    import py3Dmol

    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False


class CalculationParameters(BaseModel):
    """Paramètres spécifiques pour un calcul DFT."""
    optimize_geometry: bool = Field(False, description="Indique s'il faut effectuer une optimisation de la géométrie.")
    spin_polarized: bool = Field(False, description="Indique si le calcul doit être polarisé en spin (pour les atomes isolés ou les systèmes à électrons non appariés).")
    # On pourra ajouter plus de paramètres ici plus tard (fonctionnelle, base, etc.)


class ToolArguments(BaseModel):
    """Arguments structurés pour l'outil 'run_dft_calculation'."""
    system_description: str = Field(..., description="Description textuelle claire du système chimique, ex: 'la molécule HCN' ou 'un atome d'hydrogène isolé'.")
    calculation_params: CalculationParameters = Field(..., description="Les paramètres spécifiques du calcul DFT à effectuer.")


class GeneratedCodeFragment(BaseModel):
    """Représente un bloc de code Python généré par le LLM pour une seule étape."""
    function_code: str = Field(..., description="Le code Python complet et autonome pour une fonction de calcul (incluant les imports si nécessaire dans la fonction).")


class FinalWorkflowScript(BaseModel):
    """Le script Python final et complet généré par l'assistant."""
    summary: str = Field(..., description="Un résumé en une phrase de ce que fait le script.")
    final_code: str = Field(..., description="Le code Python complet et exécutable du workflow.")


class CalculationStep(BaseModel):
    """Représente une étape de calcul unitaire dans un plan scientifique."""
    step_id: int = Field(..., description="L'identifiant de l'étape, ex: 1")
    description: str = Field(...,
                             description="Description de l'étape pour l'utilisateur")
    tool_name: str = Field(..., description="L'outil à appeler, ex: 'run_dft_calculation'.")
    tool_args: ToolArguments = Field(..., description="Les arguments structurés pour l'outil.")


class FinalAnalysisStep(BaseModel):
    """Représente l'étape finale d'analyse mathématique."""
    description: str = Field(..., description="Description de l'analyse finale.")
    formula: str = Field(...,
                         description="La formule mathématique à appliquer si necessaire, sinon rien")


class ScientificPlan(BaseModel):
    """Un plan d'action complet pour atteindre un objectif scientifique."""
    overall_goal: str = Field(..., description="Le but général de l'utilisateur.")
    calculation_steps: List[CalculationStep] = Field(..., description="La séquence des calculs à effectuer.")
    final_analysis: Optional[FinalAnalysisStep] = Field(None,
                                                        description="L'analyse finale pour combiner les résultats.")


class AtomDefinition(BaseModel):
    """Définition d'un atome avec position."""
    element: str = Field(..., description="Symbole chimique (ex: H, C, N, O)")
    position: List[float] = Field(..., description="Coordonnées [x, y, z] en Angström")


class BigDFTMoleculeProposal(BaseModel):
    """Proposition de structure moléculaire par le LLM."""
    name: str = Field(..., description="Nom de la molécule (ex: HCN, H2O)")
    atoms: List[AtomDefinition] = Field(..., description="Liste des atomes avec positions")
    charge: int = Field(0, description="Charge totale du système")
    multiplicity: int = Field(1, description="Multiplicité de spin")
    confidence: float = Field(..., description="Niveau de confiance de la proposition (0.0-1.0)")
    explanation: str = Field(..., description="Explication de la structure proposée")
    geometry_type: str = Field("unknown", description="Type de géométrie (linear, bent, tetrahedral, etc.)")


class BigDFTVisualizationCode(BaseModel):
    """Code de visualisation 3D généré."""
    language: str = Field("python", description="Langage du code")
    code: str = Field(..., description="Code complet pour visualiser la molécule")
    explanation: str = Field(..., description="Explication du code de visualisation")
    modifiable: bool = Field(True, description="Si le code peut être modifié par l'utilisateur")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances requises")


class BigDFTStructuredResponse(BaseModel):
    """Réponse structurée complète pour BigDFT."""
    executive_summary: str = Field(..., description="Résumé de l'action effectuée")
    molecule_proposal: Optional[BigDFTMoleculeProposal] = Field(None, description="Structure moléculaire proposée")
    visualization_code: Optional[BigDFTVisualizationCode] = Field(None, description="Code de visualisation")
    next_instructions: str = Field(..., description="Instructions pour l'utilisateur")
    stage_reached: str = Field(..., description="Étape atteinte dans le workflow")


class BigDFTStage(Enum):
    """Étapes de construction d'une simulation BigDFT."""
    WELCOME = "welcome"
    DEFINE_SYSTEM = "define_system"
    STRUCTURE_CREATED = "structure_created"
    DEFINE_CALCULATION = "define_calculation"
    CALCULATION_SETUP = "calculation_setup"
    READY_TO_RUN = "ready_to_run"
    COMPLETED = "completed"
    AWAIT_PLAN_CONFIRMATION = "await_plan_confirmation"
    EXECUTING_PLAN = "executing_plan"
    AWAIT_VISUALIZATION_CONFIRMATION = "await_visualization_confirmation"


@dataclass
class MolecularSystem:
    """Représentation Python native d'un système moléculaire (sans BigDFT)."""
    name: str
    atoms: List[Dict[str, Any]]  # [{"element": "O", "position": [0,0,0]}, ...]
    charge: int = 0
    multiplicity: int = 1
    cell_parameters: Optional[Dict[str, float]] = None
    is_periodic: bool = False


@dataclass
class BigDFTCalculationConfig:
    """Configuration de calcul BigDFT (sans imports BigDFT)."""
    functional: str = "PBE"
    basis_set: str = "SZ"
    convergence_criterion: float = 1e-6
    max_iterations: int = 50
    calculate_forces: bool = True
    optimize_geometry: bool = False
    spin_polarized: bool = False


@dataclass
class BigDFTSession:
    """État de la session BigDFT."""
    stage: BigDFTStage = BigDFTStage.WELCOME
    system: Optional[MolecularSystem] = None  # BigDFT System
    calculation_config: Optional[BigDFTCalculationConfig] = None
    conversation: List[Dict[str, str]] = None
    active_plan: Optional[ScientificPlan] = None
    current_step_index: int = 0
    step_results: Dict[int, Any] = None  # Pour stocker les résultats de chaque étape
    # Un dictionnaire pour stocker les codes générés [step_id -> code_string]
    generated_codes: Dict[int, str] = None
    step_clarification_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation is None:
            self.conversation = []


class BigDFTVisualization(BaseModel):
    """Configuration pour la visualisation 3D."""
    structure_data: str = Field(..., description="Données de structure (XYZ format ou autre)")
    view_type: str = Field("ball_stick", description="Type de visualisation: ball_stick, spacefill, cartoon")
    show_bonds: bool = Field(True, description="Afficher les liaisons")
    background_color: str = Field("white", description="Couleur de fond")
    width: int = Field(600, description="Largeur de la visualisation")
    height: int = Field(400, description="Hauteur de la visualisation")


class BigDFTSystemDefinition(BaseModel):
    """Définition d'un système moléculaire pour BigDFT."""
    molecule_name: str = Field(..., description="Nom de la molécule (ex: H2O, N2, etc.)")
    atoms: List[Dict[str, Any]] = Field(default_factory=list, description="Liste des atomes avec positions")
    cell_parameters: Optional[Dict[str, float]] = Field(None, description="Paramètres de maille si périodique")
    charge: int = Field(0, description="Charge totale du système")
    multiplicity: int = Field(1, description="Multiplicité de spin")


class BigDFTCalculationSetup(BaseModel):
    """Configuration du calcul BigDFT."""
    dft_functional: str = Field("PBE", description="Fonctionnelle DFT à utiliser")
    basis_set: str = Field("SZ", description="Base atomique (SZ, DZ, TZ)")
    convergence_criterion: float = Field(1e-6, description="Critère de convergence")
    max_iterations: int = Field(50, description="Nombre max d'itérations SCF")
    calculate_forces: bool = Field(True, description="Calculer les forces")
    optimization: bool = Field(False, description="Effectuer une optimisation de géométrie")


class BigDFTNotebookAssistant:
    """Assistant BigDFT pour notebook avec accès au RAG."""

    def __init__(self, rag_system=None):
        self.rag = rag_system
        self.session = BigDFTSession()
        self.logger = logging.getLogger(__name__)

    async def start_discussion(self) -> Dict[str, Any]:
        """Démarre une nouvelle discussion BigDFT."""
        self.session = BigDFTSession()  # Reset

        welcome_msg = """
# 🚀 Assistant BigDFT - Simulation DFT Interactive

Bienvenue ! Je vais vous aider à construire votre simulation BigDFT étape par étape.

## Que pouvons-nous faire ensemble ?

1. **Définir votre système** : Molécules, cristaux, surfaces
2. **Configurer le calcul** : Fonctionnelle, base, paramètres
3. **Visualiser en 3D** : Structure interactive dans le notebook  
4. **Préparer l'exécution** : Code Python prêt pour le HPC
5. **Consulter la documentation** : Accès au RAG BigDFT

## Pour commencer
Décrivez-moi votre système moléculaire. Par exemple :
- "Je veux calculer l'énergie de la molécule H2O"
- "J'ai besoin d'optimiser la géométrie de N2" 
- "Je veux étudier une surface de graphène"
        """

        self.session.conversation.append({
            "role": "assistant",
            "content": welcome_msg
        })

        return {
            "status": "started",
            "stage": self.session.stage.value,
            "message": welcome_msg
        }

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Traite un message utilisateur.
        - Si aucun plan n'est actif, en génère un.
        - Si un plan attend confirmation, gère la réponse.
        - Si un plan est en cours, traite la commande (ex: 'ok' pour continuer).
        """

        # Cas 1 : Il n'y a pas de plan actif. C'est le début de la conversation.
        if self.session.active_plan is None:
            # On ignore les messages simples comme "ok" au début.
            if user_message.lower().strip() in ['ok', 'oui', 'yes']:
                return {
                    "status": "info",
                    "message": "Veuillez d'abord décrire votre objectif scientifique, par exemple : 'Calcule l'énergie d'atomisation de H2O'."
                }

            print("🧠 Aucun plan actif. Génération d'un nouveau plan...")
            plan = await self._generate_scientific_plan(user_message)

            if not plan:
                return {"status": "error",
                        "message": "Désolé, je n'ai pas pu élaborer de plan pour votre requête. Pouvez-vous reformuler ?"}

            # Stocker le plan dans la session et initialiser les résultats
            self.session.active_plan = plan
            self.session.step_results = {}
            self.session.current_step_index = 0
            self.session.stage = BigDFTStage.AWAIT_PLAN_CONFIRMATION  # Un nouvel état !
            self.session.generated_codes = {}
            # Présenter le plan à l'utilisateur pour qu'il le valide
            return await self._display_plan_for_confirmation(plan)

        user_response = user_message.lower().strip()

        # CAS 2 : L'utilisateur confirme le PLAN -> On montre la 1ère molécule
        if self.session.stage == BigDFTStage.AWAIT_PLAN_CONFIRMATION and user_response in ['ok', 'oui']:
            print("👍 Plan confirmé. Préparation de la première étape pour visualisation.")
            self.session.step_clarification_history = None  # On vide l'historique avant de commencer
            return await self._prepare_and_visualize_current_step()

        # CAS 3 : L'utilisateur confirme une MOLÉCULE -> On génère et stocke le code avec l'agent RAG
        elif self.session.stage == BigDFTStage.AWAIT_VISUALIZATION_CONFIRMATION and user_response in ['ok', 'oui']:
            # SOUS-CAS 3.1 : L'utilisateur VALIDE la molécule
            if user_response in ['ok', 'oui']:
                print(f"🔬 Molécule confirmée. Génération du code...")
                self.session.step_clarification_history = None  # On vide l'historique avant de passer à la suite
                return await self._generate_and_store_code_for_current_step()

            # SOUS-CAS 3.2 : L'utilisateur CORRIGE la proposition
            else:
                print(f"✏️ Correction reçue : '{user_message}'. Nouvelle tentative de génération de la molécule...")
                # On ajoute la correction de l'utilisateur à l'historique de l'étape
                if self.session.step_clarification_history:
                    self.session.step_clarification_history.append({"role": "user", "content": user_message})
                # On relance la visualisation, qui utilisera maintenant l'historique enrichi
                return await self._prepare_and_visualize_current_step()

        # Gérer les annulations ou les messages non compris
        elif user_response in ['annuler', 'non', 'stop']:
            await self.start_discussion()
            return {"status": "plan_cancelled", "message": "Plan annulé."}
        else:
            return {"status": "info", "message": "En attente de votre confirmation ('ok') pour continuer."}

    async def _generate_and_store_code_for_current_step(self) -> Dict[str, Any]:
        """
        ORCHESTRATEUR : Configure la session pour l'étape actuelle, appelle le
        générateur de code, puis gère le résultat pour faire avancer le plan.
        """
        plan = self.session.active_plan
        step = plan.calculation_steps[self.session.current_step_index]

        # 1. Configurer la session pour l'étape en cours
        params = step.tool_args.calculation_params
        # On met à jour la configuration globale de la session
        self.session.calculation_config = BigDFTCalculationConfig(
            optimize_geometry=params.optimize_geometry,
            spin_polarized=params.spin_polarized
            # NOTE : Si le plan spécifiait la fonctionnelle, on l'ajouterait ici.
            # Pour l'instant, on garde PBE/SZ par défaut.
        )

        # 2. Appeler le générateur de code centralisé
        bigdft_code = await self._generate_bigdft_code()

        # 3. Gérer le résultat
        if not bigdft_code:
            return {"status": "error", "message": "L'agent RAG n'a pas pu générer le code pour cette étape."}

        # 4. Afficher et stocker le code
        display(Markdown(f"#### ✅ Code pour l'Étape {step.step_id} Généré"))
        display(Markdown(f"```python\n{bigdft_code}\n```"))
        self.session.generated_codes[step.step_id] = bigdft_code

        # 5. Passer à l'étape suivante du plan
        self.session.current_step_index += 1
        return await self._prepare_and_visualize_current_step()

    async def _assemble_final_script_with_llm(self) -> Dict[str, Any]:
        """
        Prend tous les fragments de code, et demande au LLM de les assembler
        en UNE SEULE fonction autonome contenant des fonctions imbriquées.
        """
        plan = self.session.active_plan
        context = f"Objectif final : {plan.overall_goal}\nFormule d'analyse : {plan.final_analysis.formula}\n"
        for step_id, code in self.session.generated_codes.items():
            context += f"\n--- CODE POUR ÉTAPE {step_id} ---\n{code}\n"

        # --- PROMPT FINAL CORRIGÉ ET STRICT ---
        system_prompt = f"""
    Tu es un assembleur de code expert, spécialisé dans la préparation de scripts pour l'exécution sur des clusters HPC.
    Ta mission est de prendre plusieurs fragments de code Python et de les combiner en UNE SEULE fonction autonome et exécutable.

    RÈGLES STRICTES D'ASSEMBLAGE :
    1.  **Fonction Unique :** Le résultat final doit être une seule et unique fonction, nommée `run_complete_hpc_workflow()`. Le code ne doit rien contenir en dehors de cette fonction.
    2.  **Imports à l'Intérieur :** TOUS les `import` nécessaires (ex: `from BigDFT...`) doivent être placés au début, À L'INTÉRIEUR de la fonction `run_complete_hpc_workflow()`. Regroupe-les pour éviter les doublons.
    3.  **Fonctions Imbriquées (Nested Functions) :** Chaque fragment de code fourni doit être défini comme une fonction imbriquée (une fonction à l'intérieur d'une autre) À L'INTÉRIEUR de `run_complete_hpc_workflow()`.
    4.  **Orchestration :** Le corps principal de `run_complete_hpc_workflow()`, après les définitions des imports et des fonctions imbriquées, doit :
        a. Appeler chaque fonction d'étape dans l'ordre.
        b. Stocker les énergies retournées dans des variables.
        c. Utiliser ces variables pour calculer et imprimer le résultat de l'analyse finale.
        d. Retourner le résultat final.

    CONTEXTE FOURNI :
    {context}

    Maintenant, génère le script final en respectant scrupuleusement ces règles. Le résultat doit être uniquement le code de la fonction `run_complete_hpc_workflow()`.
    """

        try:
            # L'appel au LLM avec le modèle Pydantic reste le même, il est robuste.
            response = await self.rag.unified_agent.llm.generate_response(
                messages=[{"role": "system", "content": system_prompt}],
                pydantic_model=FinalWorkflowScript
            )
            final_script = FinalWorkflowScript.model_validate(response)

            # Réinitialiser pour la prochaine discussion
            await self.start_discussion()

            return {
                "status": "code_ready",
                "message": f"Workflow assemblé ! {final_script.summary}",
                "code": final_script.final_code,
                "config_summary": {}
            }
        except Exception as e:
            return {"status": "error", "message": f"Erreur lors de l'assemblage final structuré : {e}"}

    async def _display_plan_for_confirmation(self, plan: 'ScientificPlan') -> Dict[str, Any]:
        """Affiche le plan généré par le LLM pour validation par l'utilisateur."""

        # Construire le message Markdown
        md = f"""### 📝 Proposition de Plan d'Action

    J'ai analysé votre objectif : **"{plan.overall_goal}"**.
    Voici le plan que je propose pour y parvenir. Veuillez le vérifier avant de continuer.

    ---
    """

        # Afficher les étapes de calcul
        md += "\n#### ⚙️ **Étapes de Calcul**\n"
        if plan.calculation_steps:
            for i, step in enumerate(plan.calculation_steps):
                # ✅ CORRECTION : Ajout du retour à la ligne
                md += f"**{i + 1}. {step.description}**\n"

                # Accéder à 'calculation_params' comme un attribut d'objet
                params = step.tool_args.calculation_params

                details = []
                if params.optimize_geometry:
                    details.append("Optimisation de géométrie")
                if params.spin_polarized:
                    details.append("Spin polarisé")

                if details:
                    # ✅ CORRECTION : Ajout du retour à la ligne
                    md += f"   - *Détails : {', '.join(details)}*\n"
        else:
            md += "- Aucune étape de calcul nécessaire.\n"

        # Afficher l'étape d'analyse finale (inchangé)
        if plan.final_analysis:
            md += "\n#### 📊 **Étape d'Analyse Finale**\n"
            md += f"**{len(plan.calculation_steps) + 1}. {plan.final_analysis.description}**\n"

            readable_formula = plan.final_analysis.formula
            for i, step in enumerate(plan.calculation_steps):
                readable_formula = readable_formula.replace(f"result_{step.step_id}", f"Résultat(Étape {step.step_id})")

            md += f"   - *Formule : `{readable_formula}`*\n"

        md += """
    ---
    ### ✅ **Prêt à continuer ?**

    - Tapez `ok` ou `oui` pour lancer l'exécution de ce plan, étape par étape.
    - Tapez `annuler` pour rejeter ce plan et recommencer.
    """

        # Afficher le Markdown dans le notebook
        display(Markdown(md))

        # Retourner un statut pour la logique principale
        return {
            "status": "plan_awaits_confirmation",
            "message": "Le plan a été présenté à l'utilisateur."
        }

    async def _prepare_and_visualize_current_step(self) -> Dict[str, Any]:
        """Prépare et retourne le payload pour afficher la molécule de l'étape actuelle."""
        plan = self.session.active_plan
        step_index = self.session.current_step_index

        # Vérifier si on a fini le plan
        if step_index >= len(plan.calculation_steps):
            print("✅ Toutes les étapes ont été construites. Assemblage du script final...")
            return await self._assemble_final_script_with_llm()

        step = plan.calculation_steps[step_index]
        if self.session.step_clarification_history is None:
            display(Markdown(f"### ➡️ Étape {step.step_id}/{len(plan.calculation_steps)} : **{step.description}**"))
            # Le premier message est la demande initiale du plan
            self.session.step_clarification_history = [{"role": "user", "content": step.tool_args.system_description}]

        #display(Markdown(f"### ➡️ Étape {step.step_id}/{len(plan.calculation_steps)} : **{step.description}**"))

        # Générer la structure moléculaire et la sauvegarder dans la session
        proposed_molecule = await self._propose_molecule_with_llm(conversation_history=self.session.step_clarification_history)
        if not proposed_molecule:
            return {"status": "error",
                    "message": f"Impossible de générer la structure pour '{step.tool_args.system_description}'."}

        # On ajoute la proposition de l'agent à l'historique pour le contexte futur
        self.session.step_clarification_history.append(
            {"role": "assistant", "content": proposed_molecule.model_dump_json()})

        await self._create_molecular_system(proposed_molecule)

        # Changer l'état pour attendre la confirmation de cette molécule
        self.session.stage = BigDFTStage.AWAIT_VISUALIZATION_CONFIRMATION

        # Retourner le payload pour l'affichage
        structured_response = await self._generate_bigdft_structured_response(proposed_molecule)
        return {
            "status": "system_proposed_structured",
            "structured_response": structured_response,
            "message": f"Veuillez vérifier la structure pour l'étape {step.step_id}."
        }

    async def report_step_result(self, step_id: int, result_value: float) -> Dict[str, Any]:  # Ajout du type de retour
        """Enregistre le résultat d'une étape et retourne le payload pour la suivante."""
        print(f"✅ Résultat pour l'étape {step_id} enregistré : {result_value}")
        self.session.step_results[step_id] = result_value
        self.session.current_step_index += 1

        # On retourne directement le payload de la prochaine étape
        return await self._prepare_and_visualize_current_step()

    async def _generate_bigdft_code(self) -> Optional[str]:
        """
        Génère le code PyBigDFT en utilisant l'AGENT RAG, en lui fournissant
        les données brutes du système (JSON) et en lui laissant la responsabilité
        d'écrire le code de définition de la géométrie.
        """
        # Étape 1 : Vérifier que les informations nécessaires existent dans la session
        if not self.session.system or not self.session.calculation_config:
            self.logger.error("Tentative de génération de code sans système ou configuration.")
            return None

        # Étape 2 : Préparer les données pour le prompt de l'agent
        system = self.session.system
        config = self.session.calculation_config

        # --- APPROCHE CORRIGÉE : Fournir la géométrie sous forme de données JSON ---
        # C'est propre, non ambigu et sépare les données de la logique.
        system_data = {
            "name": system.name,
            "charge": system.charge,
            "multiplicity": system.multiplicity,
            "atoms": system.atoms
        }
        geometry_json = json.dumps(system_data, indent=4)
        # --- FIN DE LA CORRECTION ---

        # Construire les instructions de calcul (cette partie ne change pas)
        param_instructions = []
        if config.optimize_geometry:
            param_instructions.append("- Une optimisation de la géométrie DOIT être effectuée.")
        if config.spin_polarized:
            param_instructions.append(f"- Le calcul DOIT être polarisé en spin.")
        param_instructions.append(f"- La fonctionnelle DFT à utiliser est '{config.functional}'.")
        param_text = "\n        ".join(param_instructions)

        # Étape 3 : Construire le prompt final, maintenant basé sur les données
        agent_prompt = f"""
        Ta mission est d'agir en tant qu'expert programmeur PyBigDFT. Tu dois écrire une fonction Python complète, autonome et CORRECTE.

        **Données d'Entrée Obligatoires :**

        1.  **Données du Système (Format JSON) :**
            Voici les données du système moléculaire. Tu DOIS utiliser ces données pour écrire le code Python qui définit la géométrie.

            ```json
            {geometry_json}
            ```

        2.  **Instructions de Calcul :**
            Ton code doit appliquer les contraintes de calcul suivantes :
            {param_text}

        **Ton plan d'action OBLIGATOIRE :**

        1.  **ÉCRIS LE CODE DE LA GÉOMÉTRIE :** En te basant sur les données JSON ci-dessus et sur les recherches sémantiques pour connaitre la syntaxe de la construction d'un system avec pybigdft.
        2.  **CHERCHE LA SYNTAXE DU CALCUL :** Utilise tes outils (`semantic_search`) pour trouver la syntaxe correcte afin d'appliquer les instructions de calcul (comment définir la fonctionnelle, l'optimisation, etc.).
        3.  **ASSEMBLE LA FONCTION :** Combine le tout dans une seule fonction Python nommée `run_bigdft_calculation()` comprenant également les import.
        4.  **RÉPONSE FINALE STRUCTURÉE :** Ta réponse finale DOIT être un appel à `structured_final_answer` contenant exactement UN `CodeExample`.

        Commence ta mission.
        """

        print("🤖 L'agent RAG est consulté pour écrire le code à partir des données brutes...")

        # Étape 4 : Lancer l'agent RAG et traiter sa réponse (cette partie ne change pas)
        try:
            agent_response = await self.rag.unified_agent.run(agent_prompt, use_memory=False)

            if agent_response.status == "success" and agent_response.structured_answer and agent_response.structured_answer.code_examples:
                generated_code = agent_response.structured_answer.code_examples[0].code
                print("✅ L'agent RAG a généré le code autonome à partir des données.")

                sources_md = "##### Sources consultées par l'agent :\n"
                if agent_response.sources_consulted:
                    for source in agent_response.sources_consulted:
                        sources_md += f"- {source.get_citation()}\n"
                    display(Markdown(sources_md))

                return generated_code
            else:
                error_details = agent_response.error_details or "L'agent n'a pas retourné de code structuré."
                self.logger.error(f"Échec de la génération de code par l'agent RAG : {error_details}")
                display(Markdown(f"### ❌ Erreur de l'Agent RAG\n{error_details}"))
                return None

        except Exception as e:
            self.logger.error(f"Erreur critique lors de l'appel à l'agent RAG pour la génération de code : {e}",
                              exc_info=True)
            return None

    async def _perform_final_analysis(self) -> Dict[str, Any]:  # Renommée
        """Calcule le résultat final et retourne un payload pour l'affichage."""
        plan = self.session.active_plan
        results = self.session.step_results

        # ... (la logique de vérification reste la même)

        safe_context = {f"result_{step.step_id}": results[step.step_id] for step in plan.calculation_steps}

        try:
            formula = plan.final_analysis.formula
            final_value = eval(formula, {"__builtins__": {}}, safe_context)
            readable_formula = formula
            for step_id, value in safe_context.items():
                readable_formula = readable_formula.replace(step_id, f"({value:.4f})")

            # Le plan est terminé, on le marque pour réinitialisation future
            self.session.stage = BigDFTStage.COMPLETED

            # On retourne un payload avec toutes les infos pour l'affichage
            return {
                "status": "final_analysis_ready",  # Nouveau statut à gérer
                "description": plan.final_analysis.description,
                "formula": formula,
                "readable_formula": readable_formula,
                "final_result": final_value
            }

        except Exception as e:
            return {"status": "error", "message": f"Erreur lors de l'évaluation de la formule finale : {e}"}

    async def _generate_scientific_plan(self, user_message: str) -> Optional[ScientificPlan]:
        """Demande au LLM de décomposer l'objectif de l'utilisateur en un plan exécutable."""

        system_prompt = f"""
    Tu es un assistant expert en chimie computationnelle. Ton rôle est de prendre l'objectif scientifique de l'utilisateur et de le décomposer en un plan d'action structuré, étape par étape, en utilisant les outils disponibles.


    L'utilisateur demande : "{user_message}"
    """
        try:
            # On demande au LLM de remplir notre modèle Pydantic
            response = await self.rag.unified_agent.llm.generate_response(
                messages=[{"role": "system", "content": system_prompt}],
                pydantic_model=ScientificPlan
            )
            if response and isinstance(response, dict):
                return ScientificPlan.model_validate(response)
        except Exception as e:
            print(f"Erreur lors de la génération du plan : {e}")
            return None

    def _normalize_molecule_info(self, molecule_info) -> Dict[str, Any]:
        """Convertit un objet BigDFTMoleculeProposal en dictionnaire."""
        if hasattr(molecule_info, 'model_dump'):  # Objet Pydantic
            result = molecule_info.model_dump()
            result['atoms'] = [
                {"element": atom.element, "position": atom.position}
                for atom in molecule_info.atoms
            ]
            return result
        else:
            return molecule_info  # Déjà un dictionnaire

    async def _propose_molecule_with_llm(self, conversation_history: List[Dict[str, str]]) -> Optional[
        BigDFTMoleculeProposal]:
        """
        Utilise le LLM pour proposer une molécule, en se basant sur un historique de conversation
        et en s'enrichissant du contexte RAG si nécessaire.
        """
        if not conversation_history:
            return None

        # --- 1. Logique RAG : on se base sur la dernière demande de l'utilisateur ---
        rag_context = ""
        last_user_message = ""
        # On trouve la dernière intervention de l'utilisateur pour la recherche RAG
        for message in reversed(conversation_history):
            if message["role"] == "user":
                last_user_message = message["content"]
                break

        if self.rag and last_user_message:
            try:
                retriever = self.rag.unified_agent.semantic_retriever
                if len(retriever.chunks) == 0:
                    print("    🔄 Construction de l'index RAG (une seule fois)...")
                    notebook_count = retriever.build_index_from_existing_chunks(self.rag)
                    print(f"    ✅ {notebook_count} notebook(s) indexés.")

                if len(retriever.chunks) > 0:
                    print(f"    🔍 Recherche RAG pour : '{last_user_message}'")
                    rag_results = retriever.query(f"structure de la molécule {last_user_message}", k=2)
                    if rag_results:
                        rag_context = "\n".join([r["content"] for r in rag_results[:1]])
            except Exception as e:
                print(f"    ⚠️ Erreur RAG: {e}")

        # --- 2. Construction du Prompt Système ---
        # Ce prompt est maintenant un guide de comportement pour le LLM.
        system_prompt = f"""
    Tu es un expert en chimie computationnelle. Ta mission est de déterminer une structure moléculaire en te basant sur la conversation ci-dessous.

    RÈGLES FONDAMENTALES :
    1.  **PRIORITÉ À L'UTILISATEUR** : Si l'utilisateur te corrige (par exemple en disant 'non, je veux juste C'), sa demande écrase toute autre considération. Tu DOIS lui fournir ce qu'il demande, même si cela te semble physiquement inhabituel (comme un atome de carbone isolé).
    2.  **UTILISE LE CONTEXTE RAG** : Sers-toi du contexte RAG fourni comme base de connaissance pour ta première proposition si aucune correction n'est faite.
    3.  **SOIS PRÉCIS** : Tes structures doivent avoir des coordonnées en Angström et des distances de liaison réalistes, sauf si l'utilisateur demande autre chose.
    4.  **EXPLIQUE** : Justifie brièvement ta proposition dans le champ 'explanation'.

    Contexte RAG disponible :
    {rag_context if rag_context else "Aucun."}
    """

        # --- 3. Appel au LLM avec l'historique complet ---
        try:
            if self.rag and hasattr(self.rag, 'unified_agent'):
                # On combine le prompt système avec l'historique de la conversation
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(conversation_history)

                # On utilise la réponse structurée pour plus de fiabilité
                response = await self.rag.unified_agent.llm.generate_response(
                    messages=messages,
                    pydantic_model=BigDFTMoleculeProposal
                )

                if response and isinstance(response, dict):
                    molecule_proposal = BigDFTMoleculeProposal.model_validate(response)
                    print(f"    ✅ Structure proposée: {molecule_proposal.name}")
                    print(f"    📊 Confiance: {molecule_proposal.confidence:.2f}")
                    print(f"    📝 {molecule_proposal.explanation}")
                    return molecule_proposal

        except Exception as e:
            print(f"    ❌ Erreur LLM structuré: {e}")

        return None

    async def _create_and_display_system(self, molecule_info) -> Dict[str, Any]:
        """Crée le système et génère une réponse structurée complète."""

        # Normaliser molecule_info en dictionnaire pour la compatibilité
        if hasattr(molecule_info, 'model_dump'):  # C'est un objet Pydantic
            molecule_dict = molecule_info.model_dump()
            molecule_dict['atoms'] = [
                {"element": atom.element, "position": atom.position}
                for atom in molecule_info.atoms
            ]
        else:
            molecule_dict = molecule_info

        # Créer le système avec le dictionnaire
        system_created = await self._create_molecular_system(molecule_dict)
        if not system_created:
            return {"status": "error", "message": "Erreur lors de la création du système"}

        # Générer une réponse structurée (passer l'objet Pydantic original si disponible)
        structured_response = await self._generate_bigdft_structured_response(
            molecule_info if hasattr(molecule_info, 'model_dump') else molecule_dict
        )

        self.session.stage = BigDFTStage.STRUCTURE_CREATED

        return {
            "status": "system_proposed_structured",
            "structured_response": structured_response,
            "system_info": molecule_dict,  # Toujours passer le dict pour compatibilité
            "message": f"Structure proposée pour {molecule_dict['name']}"
        }

    async def _generate_bigdft_structured_response(self, molecule_info) -> BigDFTStructuredResponse:
        """Génère une réponse structurée complète avec code de visualisation."""

        # ✅ CORRECTION : Gérer à la fois les objets Pydantic et les dictionnaires
        if hasattr(molecule_info, 'model_dump'):  # C'est un objet BigDFTMoleculeProposal
            molecule_proposal = molecule_info
            # Convertir en dict pour le code de visualisation
            molecule_dict = {
                "name": molecule_proposal.name,
                "charge": molecule_proposal.charge,
                "multiplicity": molecule_proposal.multiplicity,
                "confidence": molecule_proposal.confidence,
                "explanation": molecule_proposal.explanation,
                "atoms": [
                    {"element": atom.element, "position": atom.position}
                    for atom in molecule_proposal.atoms
                ]
            }
        else:  # C'est un dictionnaire
            molecule_dict = molecule_info
            # Convertir en BigDFTMoleculeProposal
            atoms_list = []
            for atom in molecule_dict.get("atoms", []):
                atoms_list.append(AtomDefinition(
                    element=atom["element"],
                    position=atom["position"]
                ))

            molecule_proposal = BigDFTMoleculeProposal(
                name=molecule_dict["name"],
                atoms=atoms_list,
                charge=molecule_dict.get("charge", 0),
                multiplicity=molecule_dict.get("multiplicity", 1),
                confidence=molecule_dict.get("confidence", 0.9),
                explanation=molecule_dict.get("explanation", f"Structure standard pour {molecule_dict['name']}"),
                geometry_type=molecule_dict.get("geometry_type", "unknown")
            )

        # Génération du code de visualisation avec le dictionnaire
        visualization_code = self._generate_visualization_code_structured(molecule_dict)

        return BigDFTStructuredResponse(
            executive_summary=f"Structure moléculaire {molecule_proposal.name} proposée avec {len(molecule_proposal.atoms)} atomes.",
            molecule_proposal=molecule_proposal,
            visualization_code=visualization_code,
            next_instructions="Vous pouvez modifier le code de visualisation ci-dessous, puis confirmer avec 'ok' pour passer à la configuration DFT.",
            stage_reached="structure_proposed"
        )

    def _generate_visualization_code_structured(self, molecule_dict: Dict[str, Any]) -> BigDFTVisualizationCode:
        """Génère le code de visualisation sous forme structurée."""

        # ✅ CORRECTION : Utiliser molecule_dict (dictionnaire) au lieu d'objet Pydantic
        code_lines = [
            "# 🧪 Visualisation moléculaire interactive - BigDFT Assistant",
            f"# Molécule: {molecule_dict['name']}",
            "# Vous pouvez modifier les coordonnées ci-dessous",
            "",
            "import numpy as np",
            "",
            "# Structure moléculaire modifiable",
            "molecule_data = {",
            f'    "name": "{molecule_dict["name"]}",',
            f'    "charge": {molecule_dict.get("charge", 0)},',
            f'    "multiplicity": {molecule_dict.get("multiplicity", 1)},',
            '    "atoms": ['
        ]

        for atom in molecule_dict["atoms"]:
            pos = atom["position"]
            code_lines.append(
                f'        {{"element": "{atom["element"]}", "position": [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]}},')

        code_lines.extend([
            "    ]",
            "}",
            "",
            "# Génération automatique du format XYZ",
            "def generate_xyz_from_molecule(mol_data):",
            '    lines = [str(len(mol_data["atoms"]))]',
            '    lines.append(f"{mol_data[\'name\']} - Structure modifiable")',
            '    for atom in mol_data["atoms"]:',
            '        pos = atom["position"]',
            '        lines.append(f"{atom[\'element\']:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")',
            '    return "\\n".join(lines)',
            "",
            "xyz_content = generate_xyz_from_molecule(molecule_data)",
            "",
            "# Visualisation 3D avec py3Dmol",
            "try:",
            "    import py3Dmol",
            "    view = py3Dmol.view(width=700, height=500)",
            "    view.addModel(xyz_content, 'xyz')",
            "    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.4}})",
            "    view.setBackgroundColor('#f0f0f0')",
            "    view.addLabel('Modifiez les coordonnées ci-dessus puis ré-exécutez', {'position': {'x': 10, 'y': 10, 'z': 0}, 'backgroundColor': 'black', 'fontColor': 'white'})",
            "    view.zoomTo()",
            "    view.show()",
            "    ",
            "except ImportError:",
            "    print('⚠️ py3Dmol non disponible. Installation: pip install py3Dmol')",
            "    print('📊 Structure XYZ:')",
            "    print(xyz_content)",
            "",
            "# Afficher les informations moléculaires",
            'print(f"🧪 Molécule: {molecule_data[\'name\']}")',
            'print(f"⚛️ Atomes: {len(molecule_data[\'atoms\'])}")',
            'print(f"⚡ Charge: {molecule_data[\'charge\']}")',
            'print(f"🌀 Multiplicité: {molecule_data[\'multiplicity\']}")',
            'print("\\n✅ Structure prête ! Dites \'ok\' pour configurer le calcul DFT.")'
        ])

        return BigDFTVisualizationCode(
            language="python",
            code="\n".join(code_lines),
            explanation=f"Code de visualisation 3D interactif pour {molecule_dict['name']} avec py3Dmol",
            modifiable=True,
            dependencies=["py3Dmol", "numpy"]
        )

    async def _prepare_bigdft_code_for_execution(self, structured_response: BigDFTStructuredResponse):
        """Prépare la réponse BigDFT pour l'exécution via /execute."""

        if not structured_response.visualization_code:
            return

        # Créer une réponse structurée compatible avec /execute
        from OntoFlow.agent.Onto_wa_rag.jupyter_analysis.jupyter_agent import AgentStructuredAnswerArgs, CodeExample
        from OntoFlow.agent.Onto_wa_rag.jupyter_analysis.jupyter_agent import AgentResponse
        from datetime import datetime

        # Convertir le code BigDFT en CodeExample
        viz_code = structured_response.visualization_code

        agent_structured_response = AgentStructuredAnswerArgs(
            executive_summary=structured_response.executive_summary,
            code_examples=[
                CodeExample(
                    language=viz_code.language,
                    code=viz_code.code,
                    explanation=viz_code.explanation,
                    function_name="generate_xyz_from_molecule",
                    execution_ready=True,
                    is_complete_function=True,
                    required_modules=viz_code.dependencies
                )
            ],
            answer_type="hpc_function" if "BigDFT" in viz_code.code else "explanation"
        )

        # Créer une réponse d'agent simulée pour /execute
        fake_response = AgentResponse(
            answer=structured_response.executive_summary,
            status="success",
            query="BigDFT molecule visualization",
            session_id="bigdft_viz_session",
            timestamp=datetime.now(),
            execution_time_total_ms=1000.0,
            steps_taken=1,
            max_steps=1,
            structured_answer=agent_structured_response
        )

        return fake_response

    def _generate_editable_molecule_code(self, molecule_info: Dict[str, Any]) -> str:
        """Génère le code Python modifiable pour la molécule."""

        lines = [
            "# 📝 Structure moléculaire modifiable",
            f"# Molécule: {molecule_info['name']}",
            "# Modifiez les coordonnées ci-dessous si nécessaire",
            "",
            "import py3Dmol",
            "",
            "# Définition des atomes (élément, x, y, z en Angström)",
            "atoms = ["
        ]

        for atom in molecule_info["atoms"]:
            element = atom["element"]
            pos = atom["position"]
            lines.append(f'    ("{element}", {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}),')

        lines.extend([
            "]",
            "",
            "# Génération du fichier XYZ",
            "xyz_content = f\"{len(atoms)}\\n\"",
            f"xyz_content += \"{molecule_info['name']} - Structure modifiable\\n\"",
            "for element, x, y, z in atoms:",
            "    xyz_content += f\"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}\\n\"",
            "",
            "# Visualisation 3D",
            "try:",
            "    view = py3Dmol.view(width=600, height=400)",
            "    view.addModel(xyz_content, 'xyz')",
            "    view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'radius': 0.5}})",
            "    view.setBackgroundColor('white')",
            "    view.zoomTo()",
            "    view.show()",
            "except ImportError:",
            "    print('⚠️ py3Dmol non installé. Installez avec: pip install py3Dmol')",
            "    print('Structure XYZ:')",
            "    print(xyz_content)"
        ])

        return "\n".join(lines)

    async def _create_molecular_system(self, molecule_info) -> bool:
        """Crée un système moléculaire Python natif (sans BigDFT)."""
        try:
            # ✅ CORRECTION : Gérer à la fois dict et objet Pydantic
            if hasattr(molecule_info, 'model_dump'):  # Objet Pydantic
                name = molecule_info.name
                atoms = [{"element": atom.element, "position": atom.position} for atom in molecule_info.atoms]
                charge = molecule_info.charge
                multiplicity = molecule_info.multiplicity
            else:  # Dictionnaire
                name = molecule_info["name"]
                atoms = molecule_info["atoms"]
                charge = molecule_info.get("charge", 0)
                multiplicity = molecule_info.get("multiplicity", 1)

            self.session.system = MolecularSystem(
                name=name,
                atoms=atoms,
                charge=charge,
                multiplicity=multiplicity
            )
            return True

        except Exception as e:
            self.logger.error(f"Erreur création système: {e}")
            return False

    def _system_to_xyz(self) -> str:
        """Convertit le système en format XYZ."""
        if not self.session.system:
            return ""

        lines = []
        atoms = self.session.system.atoms

        # Format XYZ standard
        lines.append(str(len(atoms)))
        lines.append(f"Generated by BigDFT Assistant - {self.session.system.name}")

        for atom in atoms:
            element = atom["element"]
            pos = atom["position"]
            lines.append(f"{element:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

        return "\n".join(lines)

    async def _generate_3d_view(self) -> Optional[Dict[str, Any]]:
        """Génère une visualisation 3D du système."""
        if not self.session.system:
            return None

        try:
            # Convertir le système en format XYZ
            xyz_data = self._system_to_xyz()

            if HAS_PY3DMOL:
                return {
                    "type": "py3dmol",
                    "data": xyz_data,
                    "config": {
                        "style": "ball_stick",
                        "width": 600,
                        "height": 400
                    }
                }
            elif HAS_NGLVIEW:
                return {
                    "type": "nglview",
                    "data": xyz_data
                }
            else:
                return {
                    "type": "text",
                    "data": xyz_data,
                    "message": "Visualisation 3D non disponible. Installez py3Dmol ou nglview."
                }

        except Exception as e:
            self.logger.error(f"Erreur visualisation: {e}")
            return None

    async def _get_system_suggestions(self, user_input: str) -> List[str]:
        """Génère des suggestions pour définir le système."""
        return [
            "Molécule H2O (eau)",
            "Molécule N2 (diazote)",
            "Molécule CO2 (dioxyde de carbone)",
            "Molécule CH4 (méthane)",
            "Définir manuellement les atomes"
        ]

    def _check_rag_status(self) -> Dict[str, Any]:
        """Vérifie et retourne l'état du système RAG."""
        if not self.rag:
            return {"available": False, "reason": "RAG system not initialized"}

        if not hasattr(self.rag, 'unified_agent'):
            return {"available": False, "reason": "Unified agent not available"}

        retriever = self.rag.unified_agent.semantic_retriever
        chunks_count = len(retriever.chunks) if hasattr(retriever, 'chunks') else 0

        return {
            "available": True,
            "chunks_count": chunks_count,
            "indexed_files": len(getattr(retriever, 'indexed_files', {})),
            "needs_indexing": chunks_count == 0
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel de la session."""
        return {
            "stage": self.session.stage.value,
            "has_system": self.session.system is not None,
            "has_calculator": self.session.calculator is not None,
            "conversation_length": len(self.session.conversation)
        }