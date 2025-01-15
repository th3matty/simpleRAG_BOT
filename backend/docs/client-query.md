flowchart TD
subgraph routes_chat["routes/chat.py"]
A["POST /chat"] -->|"chat()"| B["ChatRequest"]
B -->|"process_query()"| C["LLMService"]
end

    subgraph llm_service["services/llm.py: LLMService"]
        C -->|"messages.create()"| D["Anthropic API"]
        D -->|"parse response"| E{"tool_use?"}
    end

    subgraph tool_executor["services/tools.py: ToolExecutor"]
        E -->|"Yes"| F["execute_tool()"]
        F -->|"_execute_search()"| G["get_single_embedding()"]
    end

    subgraph database["database.py: ChromaDB"]
        G -->|"query_documents()"| H["collection.query()"]
        H -->|"process results"| I["calculate_similarity()"]
    end

    subgraph response_gen["services/llm.py: generate_response"]
        I -->|"format results"| J["final LLM call"]
        E -->|"No"| J
        J -->|"format"| K["ChatResponse"]
    end

    subgraph config["config.py: Settings"]
        L["Settings"] -.->|"configure"| C
        L -.->|"configure"| H
    end

    %% Add labels for method details
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef config fill:#fcf,stroke:#333,stroke-width:1px
    class L config
